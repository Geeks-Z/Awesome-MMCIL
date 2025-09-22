# 🚀 Awesome-MMCL

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" style="zoom: 60%;" /></div>
<p></p>
<div align=center><img src="https://visitor-badge.laobi.icu/badge?page_id=Awesome-MMCL&left_color=green&right_color=red" /> <img src="https://img.shields.io/github/last-commit/your-repo/Awesome-MMCL" /> <img src="https://img.shields.io/github/license/your-repo/Awesome-MMCL" /></div>

## 🎉 Introduction

- **多模态增量学习**：Multimodal Continual Learning (MMCL)
- **汇总**多模态增量学习的资源、代码和论文，并对部分方法进行复现
- [论文阅读博客](https:www.zhihu.com/column/c_1860408728035147776)

---

## 🚀 Survey

| Title                                                        | Venue | Year | Code |
| ------------------------------------------------------------ | ----- | ---- | ---- |
| [When continue learning meets multimodal large language model: a survey](http:arxiv.org/abs/2503.01887) |       | 2025 |      |
| [Recent advances of multimodal continual learning: a comprehensive survey](http:arxiv.org/abs/2410.05352) |       | 2024 |      |
|                                                              |       |      |      |

---

## 🌟 Papers

| Title                                                        | Method       | Venue | Year | Code                                                         |
| ------------------------------------------------------------ | ------------ | ----- | ---- | ------------------------------------------------------------ |
| [Mind the gap: preserving and compensating for the modality gap in CLIP-based continual learning](http:arxiv.org/abs/2507.09118) | MG-CLIP      | ICCV  | 2025 | [MindtheGap](https:github.com/linlany/MindtheGap)            |
| [External knowledge injection for CLIP-based class-incremental learning](http:arxiv.org/abs/2503.08510) | ENGINE       | ICCV  | 2025 | [ICCV25-ENGINE-main](https:github.com/LAMDA-CL/ICCV25-ENGINE) |
| [Learning Without Forgetting for  Vision-Language Models](https:ieeexplore.ieee.org/document/10882940/) | PROOF        | TPAMI | 2025 | [LAMDA-CL/PROOF](https:github.com/LAMDA-CL/PROOF)            |
| [Class-incremental learning with CLIP: adaptive representation adjustment and parameter fusion](https:link.springer.com/10.1007/978-3-031-72949-2_13) | RAPF         | ECCV  | 2024 | [linlany/RAPF](https:github.com/linlany/RAPF)                |
| [Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters](http:arxiv.org/abs/2403.11549) | MoE-Adapters | CVPR  | 2024 | [MoE-Adapters4CL](https:github.com/JiazuoYu/MoE-Adapters4CL) |
|                                                              |              |       |      |                                                              |

---

## 📚 Datasets

| Dataset       | training instances | testing instances | Classes | Link                                                         | Abstract                                 |
| ------------- | ------------------ | ----------------- | ------- | ------------------------------------------------------------ | ---------------------------------------- |
| CIFAR100      | 50,000             | 10,000            | 100     |                                                              |                                          |
| CUB200-2011   | 9,430              | 2,358             | 200     |                                                              | 加州理工学院2010年提出的鸟类细粒度数据集 |
| ImageNet-R    | 24,000             | 6,000             | 200     |                                                              |                                          |
| ImageNet-A    | 5,981              | 1,519             | 200     |                                                              |                                          |
| ObjectNet     | 26,509             | 6,628             | 200     | [https:objectnet.dev/download.html](https:objectnet.dev/download.html) |                                          |
| Omnibenchmark | 89,697             | 5,983             | 300     |                                                              |                                          |
| VTAB          | 1,796              | 8,619             | 50      |                                                              |                                          |

---

## 📊 Reproduced Results

### Details

- ### Dataset split

  - B-$m$ Inc-$n$ ：$m$代表初始增量阶段类别数量，$n$ 代表后续每个增量阶段的类别数量
  
- ### Backbone

  | 模型                   | 训练数据                               | 发布方             | 特点                                 | 模型规模                       | 创建方式                                                     |
  | ---------------------- | -------------------------------------- | ------------------ | ------------------------------------ | ------------------------------ | ------------------------------------------------------------ |
  | **OpenAI CLIP**        | OpenAI 内部收集的 4 亿图文对数据集     | OpenAI             | 官方版本，性能稳定，但训练数据不公开 | ViT-B/32, ViT-B/16, ViT-L/14等 | `model, preprocess = clip.load("ViT-B/32", device=device)`   |
  | **OpenCLIP LAION400M** | 公开的 LAION-400M 数据集（4 亿图文对） | LAION 组织开源项目 | 完全开源，训练数据公开可获取，可复现 | 多种架构和规模选择             | `model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')` |

  > 均采用`OpenAI_CLIP`

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

### CIFAR-100

> seed=1993/2 对应了不同的类别增量顺序

**seed = 1993**

|           | B0 Inc5     | B0 Inc10     | B0 Inc20    | B50 Inc5    | B50 Inc10   |
| --------- | ------------ | ----------- | ------------ | ------------ | ------------ |
| ENGINE | 82.88   | 82.7     | 81.79    | 77.81    | 78.06     |
| PROOF | 83.51   | 83.23    | 82.45    | 78.56    | 79.36     |
| RAPF | 87.17 | 86.57  | 86.02  | 83.65  | 83.41  |
| MoE-Adapters | 84.47  | 85.19  | 86.53  | 84.08 | 84.03 |
| MG-CLIP | 86.53  | 86.85  | 87.44  | 81.77  | 83.87  |

**seed = 2**

|              | B0 Inc5 | B0 Inc10 | B0 Inc20 | B50 Inc5 | B50 Inc10 |
| ------------ | ------- | -------- | -------- | -------- | --------- |
| ENGINE       | 82.28   | 82.04    | 81.21    | 76.87    | 77.54     |
| PROOF        | 83.18   | 82.88    | 82.23    | 77.84    | 78.64     |
| RAPF         | 86.76   | 85.89    | 85.44    | 82.98    | 82.85     |
| MoE-Adapters | 83.78   | 85.18    | 85.69    | 83.16    | 83.25     |
| MG-CLIP      | 85.72   | 86.98    | 87.28    | 80.98    | 83.27     |

---

### ImagerNet-R

**seed = 1993**

|              | B0 Inc10 | B0 Inc20 | B0 Inc40 | B100 Inc10 | B100 Inc20 |
| ------------ | -------- | -------- | -------- | ---------- | ---------- |
| ENGINE       | 83.66    | 83.68    | 83.11    | 80.03      | 80.43      |
| PROOF        | 81.12    | 81.02    | 80.45    | 78.67      | 79.06      |
| RAPF         | 85.58    | 85.66    | 84.63    | 81.44      | 82.51      |
| MoE-Adapters | 85.65    | 85.88    | 85.96    | 84.29      | 84.26      |
| MG-CLIP      | 87.37    | 87.47    | 87.29    | 83.92      | 84.64      |

**seed = 2**

|              | B0 Inc10 | B0 Inc20 | B0 Inc40 | B100 Inc10 | B100 Inc20 |
| ------------ | -------- | -------- | -------- | ---------- | ---------- |
| ENGINE       | 83.66    | 83.68    | 83.11    | 80.03      | 80.43      |
| PROOF        | 81.12    | 81.02    | 80.45    | 78.67      | 79.06      |
| RAPF         | 86.09    | 85.91    | 85.09    | 82.10      | 82.67      |
| MoE-Adapters | 85.99    | 86.50    | 86.57    | 84.96      | 84.42      |
| MG-CLIP      | 87.32    | 87.61    | 86.96    | 83.77      | 84.62      |

---

### CUB200

seed = 1993

|              | B0 Inc10 | B0 Inc20 | B0 Inc40 | B100 Inc10 | B100 Inc20 |
| ------------ | -------- | -------- | -------- | ---------- | ---------- |
| ENGINE       | 84.72    | 84.72    | 83.32    | 78.24      | 79.45      |
| PROOF        | 83.17    | 82.08    | 80.67    | 78.37      | 78.61      |
| RAPF         | 74.11    | 78.5     | 81.4     | 64.58      | 71.19      |
| MoE-Adapters | 65.60    | 65.67    | 66.06    | 62.25      | 63.50      |
| MG-CLIP      | 74.97    | 77.65    | 79.19    | 71.66      | 72.82      |

**seed = 2**

|              | B0 Inc10 | B0 Inc20 | B0 Inc40 | B100 Inc10 | B100 Inc20 |
| ------------ | -------- | -------- | -------- | ---------- | ---------- |
| ENGINE       | 84.72    | 84.72    | 83.32    | 78.24      | 79.45      |
| PROOF        | 83.17    | 82.08    | 80.67    | 78.37      | 78.61      |
| RAPF         | 70.91    | 76.62    | 79.49    | 63.82      | 70.29      |
| MoE-Adapters | 61.75    | 61.83    | 60.38    | 60.42      | 60.91      |
| MG-CLIP      | 72.25    | 74.41    | 75.74    | 68.65      | 71.40      |

## 
