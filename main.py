from datasets.gba_graph import GbaDataset
from torch_geometric.loader import DataLoader
from models.SemiGPS_interface import LitSemiGPS
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer
from lightning.pytorch.tuner.tuning import Tuner
import torch
import numpy as np

from config import Config

torch.set_float32_matmul_precision('high')

zone_labels = np.fromfile('datasets/lable.bin', dtype=np.int64)

train_loader = DataLoader(GbaDataset(),batch_size=1,num_workers=1,persistent_workers=True,pin_memory=True)

model = LitSemiGPS(zone_lst=zone_labels, **Config)
logger = TensorBoardLogger("logs")


''' Training model '''
trainer = Trainer(accelerator="gpu",max_epochs=100,logger=logger)#,strategy='ddp_find_unused_parameters_true')#,precision="bf16-mixed")
# trainer.fit(model=model, train_dataloaders=train_loader)#,ckpt_path="checkpoint/...")

''' (option) Finding best initial lr '''
tuner = Tuner(trainer)
tuner.lr_find(model=model, train_dataloaders=train_loader)

''' Test/Prediction '''
# model = LitSTGCN.load_from_checkpoint("checkpoint/epoch=6-step=336.ckpt")
# model.h_state = None

# trainer = Trainer(devices=1, num_nodes=1)

# test_loader = DataLoader(SzDataset(bais=0),batch_size=1,num_workers=1,persistent_workers=True,pin_memory=True)
# h_state_list = trainer.predict(model, dataloaders=test_loader)


''' Save data'''
# file_path = 'result/node_emb.pt'
# h_state_list = torch.stack(h_state_list, dim=0)[24:,:,:]
# torch.save(h_state_list, file_path)

# h_state_list.size()