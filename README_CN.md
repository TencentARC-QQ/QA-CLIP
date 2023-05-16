[**中文说明**](README_CN.md) | [**English**](README.md)
# 项目介绍
本项目旨在提供更好的中文CLIP模型。该项目使用的训练数据均为公开可访问的图像URL及相关中文文本描述，总量达到400M。经过筛选后，我们最终使用了100M的数据进行训练。
本项目由腾讯PCG QQ-ARC联合实验室完成。
<br><br>

# 模型及实验
<span id="model_card"></span>
## 模型规模 & 下载链接
QA-CLIP目前开源3个不同规模，其模型信息和下载方式见下表：

<table border="1" width="100%">
    <tr align="center">
        <th>模型规模</th><th>下载链接</th><th>参数量</th><th>视觉侧骨架</th><th>视觉侧参数量</th><th>文本侧骨架</th><th>文本侧参数量</th><th>分辨率</th>
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

## 实验结果
针对图文检索任务，我们在[MUGE Retrieval](https://tianchi.aliyun.com/muge)、[Flickr30K-CN](https://github.com/li-xirong/cross-lingual-cap)和[COCO-CN](https://github.com/li-xirong/coco-cn)上进行了zero-shot测试。
针对图像零样本分类任务，我们在ImageNet数据集上进行了测试。测试结果见下表：


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
        <td width="120%">:star:QA-CLIP<sub>RN50</sub></td><td><b>50.5</b></td><td><b>77.4</b></td><td><b>86.1</b></td><td><b>67.1</b></td><td><b>87.9</b></td><td><b>93.2</b></td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B/16</sub></td><td>62.7</td><td>86.9</td><td>92.8</td><td>74.6</td><td>93.5</td><td>97.1</td>
    </tr>  
	<tr align="center">
        <td width="120%">:star:QA-CLIP<sub>ViT-B/16</sub></td><td><b>63.8</b></td><td><b>88.0</b></td><td><b>93.2</b></td><td><b>78.4</b></td><td><b>96.1</b></td><td><b>98.5</b></td>
    </tr> 
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>68.0</td><td>89.7</td><td>94.4</td><td>80.2</td><td>96.6</td><td>98.2</td>
    </tr> 
	<tr align="center">
        <td width="120%">AltClip<sub>ViT-L/14</sub></td><td><b>69.7</b></td><td>90.1</td><td><b>94.8</b></td><td>84.8</td><td>97.7</td><td>99.1</td>
    </tr>
	<tr align="center">
        <td width="120%">:star:QA-CLIP<sub>ViT-L/14</sub></td><td>69.3</td><td><b>90.3</b></td><td>94.7</td><td><b>85.3</b></td><td><b>97.9</b></td><td><b>99.2</b></td>
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
        <td width="120%">:star:QA-CLIP<sub>RN50</sub></td><td><b>44.0</b></td><td><b>69.9</b></td><td><b>79.5</b></td><td><b>32.4</b></td><td><b>59.5</b></td><td><b>70.3</b></td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B/16</sub></td><td>52.1</td><td>76.7</td><td>84.4</td><td>38.7</td><td>65.6</td><td>75.1</td>
    </tr>  
	<tr align="center">
        <td width="120%">:star:QA-CLIP<sub>ViT-B/16</sub></td><td><b>53.2</b></td><td><b>77.7</b></td><td><b>85.1</b></td><td><b>40.7</b></td><td><b>68.2</b></td><td><b>77.2</b></td>
    </tr> 
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>56.4</td><td>79.8</td><td>86.2</td><td>42.6</td><td>69.8</td><td>78.6</td>
    </tr> 
	<tr align="center">
        <td width="120%">AltClip<sub>ViT-L/14</sub></td><td>29.6</td><td>49.9</td><td>58.8</td><td>21.4</td><td>42.0</td><td>51.9</td>
    </tr>
	<tr align="center">
        <td width="120%">:star:QA-CLIP<sub>ViT-L/14</sub></td><td><b>57.4</b></td><td><b>81.0</b></td><td><b>87.7</b></td><td><b>45.5</b></td><td><b>73.0</b></td><td><b>81.4</b></td>
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
        <td width="120%">:star:QA-CLIP<sub>RN50</sub></td><td><b>50.1</b></td><td><b>82.5</b></td><td><b>91.7</b></td><td><b>56.7</b></td><td><b>85.2</b></td><td><b>92.9</b></td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B/16</sub></td><td>62.2</td><td>87.1</td><td>94.9</td><td>56.3</td><td>84.0</td><td>93.3</td>
    </tr>  
	<tr align="center">
        <td width="120%">:star:QA-CLIP<sub>ViT-B/16</sub></td><td><b>62.9</b></td><td><b>87.7</b></td><td><b>94.7</b></td><td><b>61.5</b></td><td><b>87.6</b></td><td><b>94.8</b></td>
    </tr> 
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>64.9</td><td>88.8</td><td>94.2</td><td>60.6</td><td>84.4</td><td>93.1</td>
    </tr> 
	<tr align="center">
        <td width="120%">AltClip<sub>ViT-L/14</sub></td><td>63.5</td><td>87.6</td><td>93.5</td><td>62.6</td><td><b>88.5</b></td><td><b>95.9</b></td>
    </tr>
	<tr align="center">
        <td width="120%">:star:QA-CLIP<sub>ViT-L/14</sub></td><td><b>65.7</b></td><td><b>90.2</b></td><td><b>95.0</b></td><td><b>64.5</b></td><td>88.3</td><td>95.1</td>
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
        <td width="120%">:star:QA-CLIP<sub>RN50</sub></td><td><b>35.5</b></td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B/16</sub></td><td>48.4</td>
    </tr>  
	<tr align="center">
        <td width="120%">:star:QA-CLIP<sub>ViT-B/16</sub></td><td><b>49.7</b></td>
    </tr> 
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>54.7</td>
    </tr>
	<tr align="center">
        <td width="120%">:star:QA-CLIP<sub>ViT-L/14</sub></td><td><b>55.8</b></td>
    </tr>
</table>
<br>

<br><br>


# 使用教程
## 安装要求
环境配置要求:

* python >= 3.6.4
* pytorch >= 1.8.0 (with torchvision >= 0.9.0)
* CUDA Version >= 10.2

安装本项目所需库
```bash
cd /yourpath/QA-CLIP-main
pip install -r requirements.txt
```

## 推理代码
```bash
export PYTHONPATH=/yourpath/QA-CLIP-main
```
推理代码示例：
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
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    image_features /= image_features.norm(dim=-1, keepdim=True) 
    text_features /= text_features.norm(dim=-1, keepdim=True)    

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
```
<br><br>

## 预测及评估

### 图文检索测试数据集下载
<b>[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)</b>项目中已经预处理好测试集，这是他们提供的下载链接：

MUGE数据：[下载链接](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/MUGE.zip)

Flickr30K-CN数据：[下载链接](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/Flickr30k-CN.zip)

另外[COCO-CN](https://github.com/li-xirong/coco-cn)数据的获取需要向原作者进行申请
### ImageNet数据集下载
原始数据请自行下载，[中文标签](http://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/ImageNet-1K/label_cn.txt)和[英文标签](http://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/ImageNet-1K/label.txt)同样由<b>[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)</b>项目提供
### 图文检索评估
图文检索评估代码可以参考如下：
```bash
split=test # 指定计算valid或test集特征
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

### ImageNet零样本分类
ImageNet零样本分类的代码参考如下
```bash
bash scripts/zeroshot_eval.sh 0 \
    ${DATAPATH} imagenet \
    ViT-B-16 RoBERTa-wwm-ext-base-chinese \
    ./pretrained_weights/QA-CLIP-base.pt
```
<br><br>
# 致谢
项目代码基于<b>[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)</b>实现，非常感谢他们优秀的开源工作。
<br><br>
