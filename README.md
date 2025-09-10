# 🚀 Awesome-MMCL

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" style="zoom: 60%;" /></div>
<p></p>
<div align=center><img src="https://visitor-badge.laobi.icu/badge?page_id=Awesome-MMCL&left_color=green&right_color=red" /> <img src="https://img.shields.io/github/last-commit/your-repo/Awesome-MMCL" /> <img src="https://img.shields.io/github/license/your-repo/Awesome-MMCL" /></div>

## 🎉 Introduction

- **多模态增量学习**：Multimodal Continual Learning (MMCL)
- **汇总**多模态增量学习的资源、代码和论文，并对部分方法进行复现
- [论文阅读博客](https://www.zhihu.com/column/c_1860408728035147776)

---

## 🚀 Survey

| Title                                                        | Venue | Year | Code |
| ------------------------------------------------------------ | ----- | ---- | ---- |
| [When continue learning meets multimodal large language model: a survey](http://arxiv.org/abs/2503.01887) |       | 2025 |      |
| [Recent advances of multimodal continual learning: a comprehensive survey](http://arxiv.org/abs/2410.05352) |       | 2024 |      |
|                                                              |       |      |      |

---

## 🌟 Papers

| Title                                                        | Method       | Venue | Year | Code                                                         |
| ------------------------------------------------------------ | ------------ | ----- | ---- | ------------------------------------------------------------ |
| [Mind the gap: preserving and compensating for the modality gap in CLIP-based continual learning](http://arxiv.org/abs/2507.09118) | MG-CLIP      | ICCV  | 2025 | [MindtheGap](https://github.com/linlany/MindtheGap)          |
| [External knowledge injection for CLIP-based class-incremental learning](http://arxiv.org/abs/2503.08510) | ENGINE       | ICCV  | 2025 | [ICCV25-ENGINE-main](https://github.com/LAMDA-CL/ICCV25-ENGINE) |
| [Learning Without Forgetting for  Vision-Language Models](https://ieeexplore.ieee.org/document/10882940/) | PROOF        | TPAMI | 2025 | [LAMDA-CL/PROOF](https://github.com/LAMDA-CL/PROOF)          |
| [Class-incremental learning with CLIP: adaptive representation adjustment and parameter fusion](https://link.springer.com/10.1007/978-3-031-72949-2_13) | RAPF         | ECCV  | 2024 | [linlany/RAPF](https://github.com/linlany/RAPF)              |
| [Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters](http://arxiv.org/abs/2403.11549) | MoE-Adapters | CVPR  | 2024 | [MoE-Adapters4CL](https://github.com/JiazuoYu/MoE-Adapters4CL) |
|                                                              |              |       |      |                                                              |

---

## 📚 Datasets

| Dataset       | training instances | testing instances | Classes | Link                                                         | Abstract                                 |
| ------------- | ------------------ | ----------------- | ------- | ------------------------------------------------------------ | ---------------------------------------- |
| CIFAR100      | 50,000             | 10,000            | 100     |                                                              |                                          |
| CUB200-2011   | 9,430              | 2,358             | 200     |                                                              | 加州理工学院2010年提出的鸟类细粒度数据集 |
| ImageNet-R    | 24,000             | 6,000             | 200     |                                                              |                                          |
| ImageNet-A    | 5,981              | 1,519             | 200     |                                                              |                                          |
| ObjectNet     | 26,509             | 6,628             | 200     | [https://objectnet.dev/download.html](https://objectnet.dev/download.html) |                                          |
| Omnibenchmark | 89,697             | 5,983             | 300     |                                                              |                                          |
| VTAB          | 1,796              | 8,619             | 50      |                                                              |                                          |

---

## 📊 Reproduced Results

### Details

- ### Dataset split

  - B-$m$ Inc-$n$ ：$m$代表初始增量阶段类别数量，$n$ 代表后续每个增量阶段的类别数量；
  - LFH(learning from half)：模型训练的初始阶段先用一半的类别进行训练，然后剩下一半的类别均匀分为 $N$ 个阶段进行训练；
  - LFS(learning from scratch)：表示所有的类别均匀地分为 $N$ 个阶段进行训练

- ### Backbone

  | 模型                   | 训练数据                               | 发布方             | 特点                                 | 模型规模                       | 创建方式                                                     |
  | ---------------------- | -------------------------------------- | ------------------ | ------------------------------------ | ------------------------------ | ------------------------------------------------------------ |
  | **OpenAI CLIP**        | OpenAI 内部收集的 4 亿图文对数据集     | OpenAI             | 官方版本，性能稳定，但训练数据不公开 | ViT-B/32, ViT-B/16, ViT-L/14等 | `model, preprocess = clip.load("ViT-B/32", device=device)`   |
  | **OpenCLIP LAION400M** | 公开的 LAION-400M 数据集（4 亿图文对） | LAION 组织开源项目 | 完全开源，训练数据公开可获取，可复现 | 多种架构和规模选择             | `model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')` |

- ### Memory

  For exemplar parameters, DER, iCaRL and FOSTER set the `fixed_memory` option to false and retain the `memory_size` of 2000 for CIFAR100, while setting `fixed_memory` option to true and retaining the `memory_per_class` of 20 for ImageNet-R. On the contrary, other models are exemplar-free.

- **Dependencies**

  - pytorch 2.2.1
  - torchvision 0.17.1
  - timm 0.6.7
  - tqdm  4.65.0
  - numpy 1.21.5
  - scipy 1.10.1
  - easydict 1.13

### Results

#### CIFAR-100

|           | Backbone | B0 Inc5     | B0 Inc10     | B0 Inc20    | B50 Inc5    | B50 Inc10   |
| --------- | ----------- | ------------ | ----------- | ------------ | ------------ | ------------ |
| ENGINE |  |  | 86.92 |  |  |  |
| PROOF | OpenCLIP_LAION400M |  | 86.76 |          |          |            |
| RAPF | OpenAI_CLIP | 86.76 | 86.09 | 85.6 | 83.06 | 82.89 |
| MoE-Adapters | OpenAI_CLIP | 84.16 | 85.2 | 85.71 |          |           |
| MG-CLIP | OpenAI_CLIP | 85.72 | 86.98 | 87.28 | 81.0 | 83.27 |

### CUB200

|              | Backbone    | B0 Inc5 | B0 Inc10 | B0 Inc20 | B100 Inc10 | B100 Inc20 |
| ------------ | ----------- | ------- | -------- | -------- | ---------- | ---------- |
| ENGINE       |             |         |          | 86.66    |            |            |
| PROOF        |             |         |          | 84.55    |            |            |
| RAPF         | OpenAI_CLIP |         |          |          |            |            |
| MoE-Adapters | OpenAI_CLIP |         |          |          |            |            |
| MG-CLIP      | OpenAI_CLIP |         |          |          |            |            |

### ImagerNet-R

|              | Backbone    | B0 Inc10 | B0 Inc20 | B0 Inc40 | B100 Inc10 | B100 Inc20 |
| ------------ | ----------- | -------- | -------- | -------- | ---------- | ---------- |
| ENGINE       |             |          | 86.22    |          |            |            |
| PROOF        |             |          | 83.56    |          |            |            |
| RAPF         | OpenAI_CLIP | 86.27    | 85.97    | 84.96    | 82.24      | 82.74      |
| MoE-Adapters | OpenAI_CLIP |          |          |          |            |            |
| MG-CLIP      | OpenAI_CLIP | 87.32    | 87.61    |          | 83.77      |            |

---

## 🤗 Acknowledgments

- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
- [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
