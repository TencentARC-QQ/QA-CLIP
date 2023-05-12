[**中文说明**](README_CN.md) | [**English**](README.md)
# Introduction
This project aims to provide a better Chinese CLIP model. The training data used in this project consists of publicly accessible image URLs and related Chinese text descriptions, totaling 400 million. After screening, we ultimately used 100 million data for training. 
This project is produced by QQ-ARC Joint Lab, Tencent PCG.
<br><br>

# Models and Results
<span id="model_card"></span>
## Model Card
QA-CLIP currently has three different open-source models of different sizes, and their model information and download links are shown in the table below:
<table border="1" width="100%">
    <tr align="center">
        <th>Model</th><th>Ckp</th><th>Params</th><th>Vision</th><th>Params of Vision</th><th>Text</th><th>Params of Text</th><th>Resolution</th>
    </tr>
    <tr align="center">
        <td>QA-CLIP<sub>RN50</sub></td><td><a href="https://huggingface.co/TencentARC/QA-CLIP/resolve/main/QA-CLIP-RN50.pt">Download</a></td><td>77M</td><td>ResNet50</td><td>38M</td><td>RBT3</td><td>39M</td><td>224</td>
    </tr>
    <tr align="center">
        <td>QA-CLIP<sub>ViT-B/16</sub></td><td><a href="https://huggingface.co/TencentARC/QA-CLIP/resolve/main/QA-CLIP-base.pt">Download</a></td><td>188M</td><td>ViT-B/16</td><td>86M</td><td>RoBERTa-wwm-Base</td><td>102M</td><td>224</td>
    </tr>
    <tr align="center">
        <td>QA-CLIP<sub>ViT-L/14</sub></td><td><a href="https://huggingface.co/TencentARC/QA-CLIP/resolve/main/QA-CLIP-large.pt">Download</a></td><td>406M</td><td>ViT-L/14</td><td>304M</td><td>RoBERTa-wwm-Base</td><td>102M</td><td>224</td>
    </tr>
</table>
<br>

## Results
We conducted zero-shot tests on [MUGE Retrieval](https://tianchi.aliyun.com/muge), [Flickr30K-CN](https://github.com/li-xirong/cross-lingual-cap), and [COCO-CN](https://github.com/li-xirong/coco-cn) datasets for image-text retrieval tasks. For the image zero-shot classification task, we tested on the ImageNet dataset. The test results are shown in the table below:

**Flickr30K-CN Zero-shot Retrieval (Official Test Set)**:
<table border="1" width="120%">
	<tr align="center">
        <th>Task</th><th colspan="3">Text-to-Image</th><th colspan="3">Image-to-Text</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>RN50</sub></td><td>48.8</td><td>76.0</td><td>84.6</td><td>60.0</td><td>85.9</td><td>92.0</td>
    </tr>
	<tr align="center">
        <td width="120%">QA-CLIP<sub>RN50</sub></td><td><b>50.5</b></td><td><b>77.4</b></td><td><b>86.1</b></td><td><b>67.1</b></td><td><b>87.9</b></td><td><b>93.2</b></td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B/16</sub></td><td>62.7</td><td>86.9</td><td>92.8</td><td>74.6</td><td>93.5</td><td>97.1</td>
    </tr>  
	<tr align="center">
        <td width="120%">QA-CLIP<sub>ViT-B/16</sub></td><td><b>63.8</b></td><td><b>88.0</b></td><td><b>93.2</b></td><td><b>78.4</b></td><td><b>96.1</b></td><td><b>98.5</b></td>
    </tr> 
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>68.0</td><td>89.7</td><td>94.4</td><td>80.2</td><td>96.6</td><td>98.2</td>
    </tr> 
	<tr align="center">
        <td width="120%">AltClip<sub>ViT-L/14</sub></td><td><b>69.7</b></td><td>90.1</td><td>94.8</td><td>84.8</td><td>97.7</td><td>99.1</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>69.3</td><td><b>90.3</b></td><td><b>94.7</b></td><td><b>85.3</b></td><td><b>97.9</b></td><td><b>99.2</b></td>
    </tr>
</table>
<br>

**MUGE Zero-shot Retrieval (Official Validation Set)**:
<table border="1" width="120%">
	<tr align="center">
        <th>Task</th><th colspan="3">Text-to-Image</th><th colspan="3">Image-to-Text</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>RN50</sub></td><td>42.6</td><td>68.5</td><td>78.0</td><td>30.0</td><td>56.2</td><td>66.9</td>
    </tr>
	<tr align="center">
        <td width="120%">QA-CLIP<sub>RN50</sub></td><td><b>44.0</b></td><td><b>69.9</b></td><td><b>79.5</b></td><td><b>32.4</b></td><td><b>59.5</b></td><td><b>70.3</b></td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B/16</sub></td><td>52.1</td><td>76.7</td><td>84.4</td><td>38.7</td><td>65.6</td><td>75.1</td>
    </tr>  
	<tr align="center">
        <td width="120%">QA-CLIP<sub>ViT-B/16</sub></td><td><b>53.2</b></td><td><b>77.7</b></td><td><b>85.1</b></td><td><b>40.7</b></td><td><b>68.2</b></td><td><b>77.2</b></td>
    </tr> 
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>56.4</td><td>79.8</td><td>86.2</td><td>42.6</td><td>69.8</td><td>78.6</td>
    </tr> 
	<tr align="center">
        <td width="120%">AltClip<sub>ViT-L/14</sub></td><td>29.6</td><td>49.9</td><td>58.8</td><td>21.4</td><td>42.0</td><td>51.9</td>
    </tr>
	<tr align="center">
        <td width="120%">QA-CLIP<sub>ViT-L/14</sub></td><td><b>57.4</b></td><td><b>81.0</b></td><td><b>87.7</b></td><td><b>45.5</b></td><td><b>73.0</b></td><td><b>81.4</b></td>
    </tr>
</table>
<br>

**COCO-CN Zero-shot Retrieval (Official Test Set)**:
<table border="1" width="120%">
	<tr align="center">
        <th>Task</th><th colspan="3">Text-to-Image</th><th colspan="3">Image-to-Text</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>RN50</sub></td><td>48.1</td><td>81.3</td><td>90.5</td><td>50.9</td><td>81.1</td><td>90.5</td>
    </tr>
	<tr align="center">
        <td width="120%">QA-CLIP<sub>RN50</sub></td><td><b>50.1</b></td><td><b>82.5</b></td><td><b>91.7</b></td><td><b>56.7</b></td><td><b>85.2</b></td><td><b>92.9</b></td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B/16</sub></td><td>62.2</td><td>87.1</td><td>94.9</td><td>56.3</td><td>84.0</td><td>93.3</td>
    </tr>  
	<tr align="center">
        <td width="120%">QA-CLIP<sub>ViT-B/16</sub></td><td><b>62.9</b></td><td><b>87.7</b></td><td><b>94.7</b></td><td><b>61.5</b></td><td><b>87.6</b></td><td><b>94.8</b></td>
    </tr> 
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>64.9</td><td>88.8</td><td>94.2</td><td>60.6</td><td>84.4</td><td>93.1</td>
    </tr> 
	<tr align="center">
        <td width="120%">AltClip<sub>ViT-L/14</sub></td><td>63.5</td><td>87.6</td><td>93.5</td><td>62.6</td><td><b>88.5</b></td><td><b>95.9</b></td>
    </tr>
	<tr align="center">
        <td width="120%">QA-CLIP<sub>ViT-L/14</sub></td><td><b>65.7</b></td><td><b>90.2</b></td><td><b>95.0</b></td><td><b>64.5</b></td><td>88.3</td><td>95.1</td>
    </tr>
</table>
<br>

**Zero-shot Image Classification on ImageNet**:
<table border="1" width="120%">
	<tr align="center">
        <th>Task</th><th colspan="1">ImageNet</th>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>RN50</sub></td><td>33.5</td>
    </tr>
	<tr align="center">
        <td width="120%">QA-CLIP<sub>RN50</sub></td><td><b>35.5</b></td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B/16</sub></td><td>48.4</td>
    </tr>  
	<tr align="center">
        <td width="120%">QA-CLIP<sub>ViT-B/16</sub></td><td><b>49.7</b></td>
    </tr> 
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>54.7</td>
    </tr>
	<tr align="center">
        <td width="120%">QA-CLIP<sub>ViT-L/14</sub></td><td><b>55.8</b></td>
    </tr>
</table>

<br><br>


# Getting Started
## Installation Requirements
Environment configuration requirements:

* python >= 3.6.4
* pytorch >= 1.8.0 (with torchvision >= 0.9.0)
* CUDA Version >= 10.2

Install required packages:
```bash
cd /yourpath/QA-CLIP-main
pip install -r requirements.txt
```
<br>
## Inference Code
```bash
export PYTHONPATH=/yourpath/QA-CLIP-main
```
Inference code example：
```python
import torch 
from PIL import Image

import clip as clip
from clip import load_from_name, available_models
print("Available models:", available_models())  
# Available models: ['ViT-B-16', 'ViT-L-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()
image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0).to(device)
text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # Normalize the features. Please use the normalized features for downstream tasks.
    image_features /= image_features.norm(dim=-1, keepdim=True) 
    text_features /= text_features.norm(dim=-1, keepdim=True)    

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
```
<br>

## Prediction and Evaluation

### Download Image-text Retrieval Test Dataset
In Project <b>[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)</b>, the test set has already been preprocessed. Here is the download link they provided:

MUGE dataset：[download link](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/MUGE.zip)

Flickr30K-CN dataset：[download link](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/Flickr30k-CN.zip)

Additionally, obtaining the [COCO-CN](https://github.com/li-xirong/coco-cn) dataset requires applying to the original author.

### Download ImageNet Dataset
Please download the raw data yourself，[Chinese Label](http://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/ImageNet-1K/label_cn.txt) and [English Label](http://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/ImageNet-1K/label.txt) are provided by Project <b>[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)</b>
### Image-text Retrieval Evaluation
The image-text retrieval evaluation code can be referred to as follows:
```bash
split=test # Designate the computation of features for the valid or test set
resume=your_ckp_path
DATAPATH=your_DATAPATH
dataset_name=Flickr30k-CN
# dataset_name=MUGE

python -u eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/${dataset_name}/lmdb/${split}/imgs" \
    --text-data="${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=52 \
    --resume=${resume} \
    --vision-model=ViT-B-16 \
    --text-model=RoBERTa-wwm-ext-base-chinese

python -u eval/make_topk_predictions.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl"

python -u eval/make_topk_predictions_tr.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl"

python eval/evaluation.py \
    ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/output1.json
cat  ${DATAPATH}/datasets/${dataset_name}/output1.json

python eval/transform_ir_annotation_to_tr.py \
    --input ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl

python eval/evaluation_tr.py \
    ${DATAPATH}/datasets/${dataset_name}/${split}_texts.tr.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/output2.json
cat ${DATAPATH}/datasets/${dataset_name}/output2.json
```

### ImageNet Zero-shot Classification
The ImageNet zero-shot classification code can be referred to as follows
```bash
bash scripts/zeroshot_eval.sh 0 \
    ${DATAPATH} imagenet \
    ViT-B-16 RoBERTa-wwm-ext-base-chinese \
    ./pretrained_weights/QA-CLIP-base.pt
```
<br><br>
# Acknowledgments
The project code is based on implementation of <b>[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)</b>, and we are very grateful for their outstanding open-source contributions.
<br><br>
