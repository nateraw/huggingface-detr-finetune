import pytorch_lightning as pl

from data import CocoDataModule
from model import DetrFinetuner

if __name__ == "__main__":
    pl.seed_everything(42)
    dm = CocoDataModule()
    model = DetrFinetuner()
    trainer = pl.Trainer(gpus=1, precision=16, max_steps=300, gradient_clip_val=0.1)
    trainer.fit(model, dm)
    model.save_pretrained("./balloon_model")
    dm.feature_extractor.save_pretrained("./balloon_model")
