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


if __name__ == '__main__':
    model = LitSemiGPS(zone_lst=zone_labels, **Config)
    logger = TensorBoardLogger(save_dir="" ,name="logs")
    ''' Training model '''
    # trainer = Trainer(accelerator="gpu",devices=1,max_epochs=10000,logger=logger)#,strategy='ddp_find_unused_parameters_true')#,precision="bf16-mixed")
    # trainer.fit(model=model, train_dataloaders=train_loader)#,ckpt_path="checkpoint/...")

    ''' (option) Finding best initial lr '''
    # tuner = Tuner(trainer)
    # tuner.lr_find(model=model, train_dataloaders=train_loader)

    ''' Test/Prediction '''
    model = LitSemiGPS.load_from_checkpoint("checkpoints/epoch=1536-step=1537.ckpt")
    trainer = Trainer(devices=1, num_nodes=1, logger=logger)
    test_loader = DataLoader(GbaDataset(),batch_size=1,num_workers=1,persistent_workers=True,pin_memory=True)
    y_pred = trainer.predict(model, dataloaders=test_loader)[0]
    np.save('datasets/pred_gdp.npy',y_pred.numpy())
    # print(y_pred)
