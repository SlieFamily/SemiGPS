import lightning as pl
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs

from gps_layer import GPSModel


class LitSemiGPS(pl.LightningModule):
    
    def __init__(self, node_features, out_features, lr, temperature=1, K=2, probability=0.2, **kargs):
        super(LitSemiGPS, self).__init__()
        self.encoder = GPSModel(node_features, out_features)
        self.extraMLP = torch.nn.Sequential(
                        torch.nn.ReLU(),
                        torch.nn.Linear(out_features, 32)
        )

        self.h_state = [None, None, None]
        self.p = probability
        self.t = temperature
        self.save_hyperparameters()

    def forward(self, 
                x1, edge1_index, edge1_weight, h1_state,
                x2, edge2_index, edge2_weight, h2_state):
        
        pass


    def mask_augment(self, data):
        ''' 通过随机mask边信息得到正样本 '''
        num_edges = data.edge_index.size(1)
        mask = torch.rand(num_edges) > self.p
        edge_index = data.edge_index[:,mask].to(self.device)
        edge_attr = data.edge_attr[mask].to(self.device)
        return edge_index, edge_attr


    def node_corruption(self, x):
        ''' 通过打乱初始特征来生成负样本 '''
        return x[torch.randperm(x.size(0))].to(self.device)

    def loss_function(self, y_query, y_pos, y_neg):
        ''' InfoNCE for contrastive learning '''
        N,C = y_query.size()
        l_pos = torch.bmm(y_query.view(N,1,C), y_pos.view(N,C,1)).squeeze(-1) # 每个query对每个正样本key的点积, N x 1
        l_neg = torch.mm(y_query, y_neg.T) # 每个query对queue内的负样本点积, N x N
        logits = torch.cat([l_pos, l_neg], dim=1) # 样本集 N x (1+N) 且仅有第一列是正样本的代表  
        labels = torch.zeros(N).long().to(self.device) # 指定分母为每行的 0-th 个值
        loss = F.cross_entropy(logits/self.t, labels)

        return loss

    def training_step(self, batch, batch_idx):
        data = batch.to(self.device)
        cor_x = self.node_corruption(data.x)
        pos_edge, pos_wight = self.mask_augment(data)
        
        h = self.encoder(data.x, data.edge_index, data.edge_attr, self.h_state[0])
        y_query = self.extraMLP(h)
        y_pos, y_neg, h_pos, h_neg = self(data.x, pos_edge, pos_wight, self.h_state[1],
                                          cor_x, data.edge_index, data.edge_attr, self.h_state[2])

        loss = self.loss_function(y_query, y_pos, y_neg)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.h_state = [h.detach(), h_pos.detach(), h_neg.detach()]
        return loss
    
    def on_train_epoch_end(self):
        # 在每个epoch结束时重置隐状态
        self.h_state = [None, None, None]
        torch.cuda.empty_cache()
    

    def on_test_epoch_end(self):
        torch.cuda.empty_cache()

    def predict_step(self, batch, batch_idx):
        data = batch.to(self.device)
        h = self.encoder(data.x, data.edge_index, data.edge_attr, self.h_state)
        self.h_state = h.detach()
        return h.clone().cpu().detach()

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