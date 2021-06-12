import pytorch_lightning as pl

from data import CocoDataModule
from model import DetrFinetuner


if __name__=='__main__':
    pl.seed_everything(42)
    dm = CocoDataModule(train_batch_size=2)
    model = DetrFinetuner()
    trainer = pl.Trainer(gpus=1, precision=16, max_steps=300, gradient_clip_val=0.1)
    trainer.fit(model, dm)
