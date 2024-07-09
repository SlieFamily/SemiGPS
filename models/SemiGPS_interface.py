import lightning as pl
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)

from utils.laplace_pos_encoder import LapPENodeEncoder
from utils.kernel_pos_encoder import RWSENodeEncoder

from models.gps_layer import GPSLayer

EPS = 1e-15

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
        node_enc_lst (string): Node encoding model list (e.g. LapPE, RWSE)
        edge_enc_lst (string): Edge encoding model list
        node_encoder_bn (bool): True if need normalization after encoding node 
        LapPE_cfg (dict) & RWSE_cfg (dict):
            dim_pe: Size of Laplace PE embeddings
            dim_emb: Size of final node embedding
            model_type: LapPE(`Transformer` or `DeepSet`); RWSE(`mlp` or `linear`)
            n_layers: Num. layers in PE encoder model
            post_n_layers: Num. layers to apply after pooling
            max_freqs: Num. eigenvectors (frequencies)
            ksteps: RWSE(`range(1,21)`)
            norm_type: Raw PE normalization layer type (`BatchNorm` or `None`)
            
    """
    def __init__(self, dim_in, node_enc_lst=[], edge_enc_lst=[], node_encoder_bn=True, LapPE_cfg={}, RWSE_cfg={}):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in

        node_encoders = []
        if node_enc_lst:
            # Encode integer node features via nn.Embeddings
            for encoder_name in node_enc_lst:
                if encoder_name == 'LapPE':
                    NodeEncoder = LapPENodeEncoder(self.dim_in, **LapPE_cfg).to(torch.device("cuda:0"))
                    # Update dim_in to reflect the new dimension of the node features
                    self.dim_in = LapPE_cfg['dim_emb'] 
                elif encoder_name == 'RWSE':
                    NodeEncoder = RWSENodeEncoder(self.dim_in, **RWSE_cfg)
                    self.dim_in = RWSE_cfg['dim_emb']
                else:
                    raise ValueError(f"Unexpected PE encoding model: {encoder_name}")
                node_encoders.append(NodeEncoder)
            self.node_encoder = torch.nn.Sequential(*node_encoders)

            if node_encoder_bn:
                self.node_encoder_bn =  BatchNorm1dNode(new_layer_config(self.dim_in, -1, -1, 
                                                                has_act=False, has_bias=False, cfg=cfg))

        if edge_enc_lst:
            # Waiting for implementing, 
            # see https://github.com/rampasek/GraphGPS/blob/main/graphgps/network/gps_model.py
            pass 

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class Node2District(torch.nn.Module):
    """
    Aggregate node features to district-level feature

    Args:
        dim_in (int): Input node feature dimension
        dim_out (int): Output district feature dimension
        zone_lst (list/1darray/1Dtensor): The node label list that describes to which district one node belongs
    """
    def __init__(self, dim_in, dim_out, zone_lst):
        super(Node2District, self).__init__()
        self.mlp = torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Linear(dim_in, 2*dim_in),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2*dim_in, dim_out)
        )
        self.zone_lst = zone_lst

    def forward(self, x):
        head_emb = x.new_zeros((self.zone_lst.max()+1, x.size()[1])) # x.size([1]) is the count of nodes
        for index in range(self.zone_lst.max() + 1):
            node_index_in_dis = (self.zone_lst == index).nonzero()[0]
            head_emb[index] = torch.sum(x[node_index_in_dis], dim=0, keepdim=True)
        head_emb = self.mlp(head_emb)
        return head_emb


class LitSemiGPS(pl.LightningModule):
    
    def __init__(self, dim_in, dim_out, node_enc_lst, edge_enc_lst, zone_lst, LapPE_cfg, RWSE_cfg, 
                 local_gnn_type, global_model_type, alpha,
                 gt_dim, gt_heads, act, gps_layers,
                 dropout, attn_dropout, layer_norm, batch_norm,  **kargs):
        super(LitSemiGPS, self).__init__()

        self.alpha = alpha
        self.zone_lst = zone_lst
        self.encoder = FeatureEncoder(dim_in, node_enc_lst, edge_enc_lst, LapPE_cfg=LapPE_cfg, RWSE_cfg=RWSE_cfg)
        self.aggregator = Node2District(gt_dim, gt_dim, zone_lst)
        self.weight_node2head = torch.nn.Parameter(torch.Tensor(gt_dim, gt_dim))

        assert gt_dim == self.encoder.dim_in, "(dim_emb != gt_dim) FeatureEmbedding dimension must match GTs input dimension"

        layers = []
        for _ in range(gps_layers):
            layers.append(GPSLayer(
                dim_h=gt_dim,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=gt_heads,
                act=act,
                # pna_degrees=pna_degrees,
                # equivstable_pe=kargs['equivstable_pe'],
                dropout=dropout,
                attn_dropout=attn_dropout,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
                # bigbird_cfg=kargs['bigbird_cfg'],
                # log_attn_weights=cfg.train.mode == 'log-attn-weights',
            ))
        self.gps_module = torch.nn.Sequential(*layers)

        self.MLPhead = torch.nn.Sequential(
                        torch.nn.ReLU(),
                        torch.nn.Linear(gt_dim, gt_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(gt_dim, dim_out)
        )


        self.save_hyperparameters()

    def forward(self, batch):
        batch = self.encoder(batch)
        batch = self.gps_module(batch)
        map = self.aggregator(batch.x)
        head = self.MLPhead(map)
        return batch.x, map, head

    def corruption(self, batch):
        ''' shuffle node features and randly mask edges '''
        new_batch = batch.clone().to(self.device)
        new_batch.x = batch.x[torch.randperm(batch.x.size(0))]
        
        num_edges = batch.edge_index.size(1)
        mask = torch.rand(num_edges) > 0.3
        new_batch.edge_index = batch.edge_index[:,mask]
        new_batch.edge_weight = batch.edge_weight[mask]
        return new_batch

    def discriminate_node2district(self, node_embs, head_embs, sigmoid=True):
        values = []
        for node_id, head_id in enumerate(self.zone_lst):
            node = node_embs[node_id]
            head = head_embs[head_id]
            value = torch.matmul(node, torch.matmul(self.weight_node2head, head))
            values.append(value)
        values = torch.stack(values)
        return torch.sigmoid(values) if sigmoid else values

    def loss_function(self, x, cor_x, map, head, y):
        ''' MSE + GraphInforMax Loss for semi-supervised learning '''
        positive_loss = -torch.log(
            self.discriminate_node2district(x, map, sigmoid=True) + EPS).mean()
        negative_loss = -torch.log(
            1-self.discriminate_node2district(cor_x, map, sigmoid=True) + EPS).mean()
        infomax_loss = positive_loss + negative_loss
        mse_loss = F.mse_loss(head, y)
        return self.alpha*infomax_loss + (1-self.alpha)*mse_loss

    def training_step(self, batch, batch_idx):
        batch = batch.to(self.device)
        cor_batch = self.corruption(batch)
        
        x, map, head = self(batch)
        cor_x, _, _ = self(cor_batch)

        loss = self.loss_function(x, cor_x, map, head, batch.y)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
    

    def on_test_epoch_end(self):
        torch.cuda.empty_cache()

    def predict_step(self, batch, batch_idx):
        batch = batch.to(self.device)
        x, map, head = self(batch)
        return head.clone().cpu().detach()

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if not hasattr(self.hparams, 'lr_scheduler'):
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    # Learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if hasattr(self.hparams, 'warmup_step'):
            w_step = self.hparams.warmup_step
            if self.trainer.global_step < w_step:
                lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(w_step))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.hparams.lr
            
        self.log('lr',optimizer.param_groups[0]["lr"],on_epoch=True,on_step=True, prog_bar=True, logger=True)