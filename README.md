# 🚀 Awesome-MMCL

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" style="zoom: 60%;" /></div>
<p></p>
<div align=center><img src="https://visitor-badge.laobi.icu/badge?page_id=Awesome-MMCL&left_color=green&right_color=red" /> <img src="https://img.shields.io/github/last-commit/your-repo/Awesome-MMCL" /> <img src="https://img.shields.io/github/license/your-repo/Awesome-MMCL" /></div>

## 🎉 Introduction

- **多模态增量学习**：Multimodal Continual Learning (MMCL)

- **汇总**多模态增量学习的资源、代码和论文，并对部分方法进行复现

---

## 🚀 Survey

| Title                                                        | Venue | Year | Code |
| ------------------------------------------------------------ | ----- | ---- | ---- |
| [When continue learning meets multimodal large language model: a survey](http://arxiv.org/abs/2503.01887) |       | 2025 |      |
| [Recent advances of multimodal continual learning: a comprehensive survey](http://arxiv.org/abs/2410.05352) |       | 2024 |      |
|                                                              |       |      |      |

---

## 🌟 Papers

| Title | Method | Venue | Year | Type | Code |
| ----- | ------ | ----- | ---- | ---- | ---- |
|       |        |       |      |      |      |
|       |        |       |      |      |      |
|       |        |       |      |      |      |

---

## 📚 Datasets

| Dataset            | training instances | testing instances | Classes | Link                                                         | Abstract                                 |
| ------------------ | ------------------ | ----------------- | ------- | ------------------------------------------------------------ | ---------------------------------------- |
| CIFAR100           | 50,000             | 10,000            | 100     |                                                              |                                          |
| CUB（CUB200-2011） | 9,430              | 2,358             | 200     |                                                              | 加州理工学院2010年提出的鸟类细粒度数据集 |
| ImageNet-R         | 24,000             | 6,000             | 200     |                                                              |                                          |
| ImageNet-A         | 5,981              | 1,519             | 200     |                                                              |                                          |
| ObjectNet          | 26,509             | 6,628             | 200     | [https://objectnet.dev/download.html](https://objectnet.dev/download.html) |                                          |
| Omnibenchmark      | 89,697             | 5,983             | 300     |                                                              |                                          |
| VTAB               | 1,796              | 8,619             | 50      |                                                              |                                          |

---

## 📊 Reproduced Results

### Details

- ### Dataset split

  - `B-$m$ Inc-$n$' ：$m$代表初始增量阶段类别数量，$n$ 代表后续每个增量阶段的类别数量；
  - LFH(learning from half)，表示在模型训练的初始阶段先用一半的类别进行训练，然后剩下一半的类别均匀分为 $N$ 个阶段进行训练；
  - LFS(learning from scratch)，表示所有的类别均匀地分为 $N$ 个阶段进行训练

- ### Backbone:`ViT-B/16-IN21K`

- ### Memory

  For exemplar parameters, DER, iCaRL and FOSTER set the `fixed_memory` option to false and retain the `memory_size` of 2000 for CIFAR100, while setting `fixed_memory` option to true and retaining the `memory_per_class` of 20 for ImageNet-R. On the contrary, other models are exemplar-free.

- **Dependencies**

  - pytorch 2.0.1
  - torchvision 0.15.2
  - timm 0.6.12
  - tqdm  4.65.0
  - numpy 1.21.5
  - scipy 1.10.1
  - easydict 1.13

### Results

> 实验结果：平均准确率（Accuracy）± 标准差（Std）/ 原文结果

#### CIFAR-100

|           | B0 Inc5     | B0 Inc10     | B0 Inc20    | B50 Inc5    | B100 Inc10   |
| --------- | ----------- | ------------ | ----------- | ------------ | ------------ |
|      |         |          |          |          |            |
|      |         |          |          |          |            |
|      |         |          |          |          |            |
|      |         |          |          |          |            |

---

## 👨‍🏫 TODO

| Title | Venue | Year | Type | Code |
| ----- | ----- | ---- | ---- | ---- |
|       |       |      |      |      |
|       |       |      |      |      |
|       |       |      |      |      |
|       |       |      |      |      |

### Different PTMs

| PTM             | Pre-Trained Dataset               | Finetuned Dataset          | Description                                                                 |
| --------------- | --------------------------------- | -------------------------- | --------------------------------------------------------------------------- |
| ViT-B/16-IN1K   | ImageNet21K                      | ImageNet1K                | Vision Transformer trained on ImageNet21K and fine-tuned on ImageNet1K.    |
| ViT-B/16-IN21K  | ImageNet21K                      | ImageNet1K                | Vision Transformer trained on ImageNet21K without fine-tuning.             |
| ViT-L/16-IN1K   | ImageNet21K                      | ImageNet1K                | Large Vision Transformer trained on ImageNet21K and fine-tuned on ImageNet1K. |
| ViT-B/16-DINO   | ImageNet                         | ImageNet1K                | Self-supervised Vision Transformer trained with DINO on ImageNet.          |
| ViT-B/16-SAM    | SA-1B (Segment Anything Dataset) | COCO, ADE20K              | Vision Transformer trained on a large-scale segmentation dataset.          |
| ViT-B/16-MAE    | ImageNet21K                      | ImageNet1K                | Vision Transformer trained with Masked Autoencoder on ImageNet21K.         |
| ViT-B/16-CLIP   | OpenAI CLIP Dataset              | COCO, Flickr30K           | Vision Transformer trained on a large corpus of text-image pairs by OpenAI.|
| ResNet18/50/152 | ImageNet1K                       | CIFAR-10, CIFAR-100       | ResNet models trained on ImageNet1K.                                       |



---

## 🤗 Acknowledgments

- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
- [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
