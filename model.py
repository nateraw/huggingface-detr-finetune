import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
import torch

class DetrFinetuner(pl.LightningModule):

    def __init__(self, model_name_or_path: str = 'facebook/detr-resnet-50', num_labels: int = 1, lr: float = 1e-4, lr_backbone: float = 1e-5, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        # HACK - This block should be fixed in transformers repo to work in 1 line of code...
        model = DetrForObjectDetection.from_pretrained(self.hparams.model_name_or_path)
        state_dict = model.state_dict()
        del state_dict["class_labels_classifier.weight"]
        del state_dict["class_labels_classifier.bias"]
        config = DetrConfig.from_pretrained(self.hparams.model_name_or_path, num_labels=self.hparams.num_labels)
        model = DetrForObjectDetection(config)
        model.load_state_dict(state_dict, strict=False)

        self.model = model
        self.forward = self.model.forward

    def shared_step(self, batch, mode='train'):
        batch['labels'] = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self(**batch)
        self.log(f"{mode}_loss", outputs.loss)
        self.log_dict({f"{mode}_{k}": v for k, v in outputs.loss_dict.items()})
        return outputs.loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.hparams.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
