import torch
import pytorch_lightning as pl
from models.model import AffinityLightningModule
from config import params
from dataset.dataset import testFpDataModule

if __name__ == '__main__':
    dm = testFpDataModule(**params)
    model = AffinityLightningModule(**params)
    trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=200)
    trainer.fit(model, datamodule=dm)
