# huggingface-detr-finetune


Download and unzip balloon dataset

```
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
unzip balloon_dataset.zip
```

Convert balloon dataset from VIA format to COCO format

```
python prepare_balloon_dataset.py
```


Train DETR on the balloon dataset

```
python main.py
```
Edit of [this example](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR) by @NielsRogge.

