import os

import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from transformers import DetrFeatureExtractor


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "custom_train.json" if train else "custom_val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        # remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze()
        # remove batch dimension
        target = encoding["target"][0]

        return pixel_values, target


class CocoCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        batch["pixel_mask"] = encoding["pixel_mask"]
        batch["labels"] = labels
        return batch


class CocoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./balloon",
        model_name_or_path: str = "facebook/detr-resnet-50",
        train_batch_size: int = 4,
        val_batch_size: int = 2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.model_name_or_path = model_name_or_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(self.model_name_or_path)
        self.data_collator = CocoCollator(self.feature_extractor)
        self.train_dataset = CocoDetection(
            img_folder=os.path.join(self.data_dir, "train"), feature_extractor=self.feature_extractor
        )
        self.val_dataset = CocoDetection(
            img_folder=os.path.join(self.data_dir, "val"), feature_extractor=self.feature_extractor, train=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, collate_fn=self.data_collator, batch_size=self.train_batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=self.data_collator, batch_size=self.val_batch_size)


if __name__ == "__main__":
    dm = CocoDataModule()
