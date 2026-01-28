# 🚀 Awesome-MMCL

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" style="zoom: 60%;" /></div>
<p></p>
<div align=center><img src="https://visitor-badge.laobi.icu/badge?page_id=Awesome-MMCL&left_color=green&right_color=red" /> <img src="https://img.shields.io/github/last-commit/Geeks-Z/Awesome-MMCL" /> <img src="https://img.shields.io/github/license/Geeks-Z/Awesome-MMCL" /></div>

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
| [Continual learning for VLMs: a survey and taxonomy beyond forgetting](http://arxiv.org/abs/2508.04227) |       | 2025 |      |

---

## 🌟 Papers

| Title                                                        | Method       | Venue | Year | Code                                                         | 论文解读                                         |
| ------------------------------------------------------------ | ------------ | ----- | ---- | ------------------------------------------------------------ | ------------------------------------------------ |
| [LADA: Scalable Label-Specific CLIP Adapter for Continual Learning](https://arxiv.org/abs/2505.23271) | LADA         | ICML  | 2025 | [LADA](https://github.com/MaolinLuo/LADA)                    | https://zhuanlan.zhihu.com/p/1959602781133439861 |
| [Mind the gap: preserving and compensating for the modality gap in CLIP-based continual learning](http:arxiv.org/abs/2507.09118) | MG-CLIP      | ICCV  | 2025 | [MindtheGap](https:github.com/linlany/MindtheGap)            | https://zhuanlan.zhihu.com/p/1973063939102360605 |
| [External knowledge injection for CLIP-based class-incremental learning](http:arxiv.org/abs/2503.08510) | ENGINE       | ICCV  | 2025 | [ICCV25-ENGINE-main](https:github.com/LAMDA-CL/ICCV25-ENGINE) | https://zhuanlan.zhihu.com/p/1948401722180474817 |
| [Learning Without Forgetting for  Vision-Language Models](https:ieeexplore.ieee.org/document/10882940/) | PROOF        | TPAMI | 2025 | [LAMDA-CL/PROOF](https:github.com/LAMDA-CL/PROOF)            | https://zhuanlan.zhihu.com/p/1959573532569768109 |
| [Class-incremental learning with CLIP: adaptive representation adjustment and parameter fusion](https:link.springer.com/10.1007/978-3-031-72949-2_13) | RAPF         | ECCV  | 2024 | [linlany/RAPF](https:github.com/linlany/RAPF)                | https://zhuanlan.zhihu.com/p/1959573948623725449 |
| [Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters](http:arxiv.org/abs/2403.11549) | MoE-Adapters | CVPR  | 2024 | [MoE-Adapters4CL](https:github.com/JiazuoYu/MoE-Adapters4CL) | https://zhuanlan.zhihu.com/p/1915756018090119546 |
| [PromptFusion: Decoupling Stability and Plasticity for Continual Learning](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/2303.07223) | PromptFusion | ECCV  | 2024 | [PromptFusion](https://github.com/HaoranChen/PromptFusion)   | https://zhuanlan.zhihu.com/p/1977048983483478476 |

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

## 📊 评估指标

> - [Continual learning for VLMs: a survey and taxonomy beyond forgetting](http://arxiv.org/abs/2508.04227)
> - [Recent advances of multimodal continual learning: a comprehensive survey](http://arxiv.org/abs/2410.05352)


<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20251203104009.png" style="zoom: 60%;" /></div>

> 记 $p^j_i$ 为“模型在完成第 $i$ 个任务训练后（$i=0$ 表示预训练状态），在第 $j$ 个任务上的性能/准确率”。



**Average Accuracy** remains fundamental, measuring mean task performance over all learning stages to gauge overall proficiency.

> For conventional continual learning, only scores above the diagonal are meaningful, since they cannot give zero-shot predictions on unseen tasks. $\bar{\mathcal{A}} = \frac{1}{T}\sum_{i=1}^{T}\mathcal( \frac{1}{i} \sum_{j=1}^{i}p_i^j)$

$$
Avg=\frac{1}{T} \sum_{t=1}^{T} \left(\frac{1}{T} \sum_{i=1}^{T} p_t^{(i)}\right)
$$

**Last Accuracy** evaluates the model’s retained competence after full training, reflecting practical deployability.
$$
\textbf{Last} = \frac{1}{T} \sum_{t=1}^{T}p^t_T
$$

**Forgetting Ratio** quantifies the maximum performance drop per task post-initial learning.
$$
\textbf{Forget}=\frac{1}{T-1} \sum_{i=1}^{T-1} \max_{i \leq j \leq T-1}\left( p_j^{i} - p_T^{i} \right)
$$

**Backward Transfer (BWT)** assesses improvements or regressions on earlier tasks induced by later learning.

> <div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20260127095257.png" style="zoom: 80%;" /></div>
> 
> 就是使用模型在学习最后一个（第 $T$ 个）task以后对之前第 $i$ 个task的表现减去刚刚学完第i个时候的表现的差值(通常为负数，上图中的蓝色部分)。反应的是记忆能力！
> - 如果这个差值很大，就意味着模型对于之前学会的知识忘记的很多；
> - 如果这个差值很小，就意味着模型对于之前学会的知识忘记的很少；
> - 如果这个差值大于零，就意味着以后学到的知识对于之前学过的是一个促进的作用。

$$
\textbf{BWT}=\frac{1}{T-1} \sum_{t=1}^{T-1} \left(p_t^{(T)} -p_t^{(t)}  \right)
$$

**Forward Transfer(FWT)** 

> <div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20260127095811.png" style="zoom: 80%;" /></div>
>
> 我们以最后一个任务task $T$ 为例，这个参数反应的是机器还没有学习 $T$，只是学习了 $T$ 之前别的task时候对于 $T$ 的影响。我们用机器学习之前 $T-1$ 个task时候对任务task $T$ 的表现减去初始化参数时候对task $T$ 的表现(下图绿色部分)。用这个参数衡量机器学习之前任务对现在任务的影响。如果这个参数是正的，那么就说明这个机器是会触类旁通的。反应的是**迁移能力**！
> $$
> FWT = \frac{1}{T-1}\sum_{i=2}^TR_{i-1,i}-R_{0,i}
> $$

**Zero-shot Transfer** evaluates generalization to unseen tasks using pre-acquired knowledge.
$$
\textbf{Transfer} =\frac{1}{T-1} \sum_{t=2}^{T} \left(\frac{1}{t-1} \sum_{i=1}^{t-1} p_t^{(i)}\right)
$$



**Zero-Shot Degradation** explicitly measures erosion of this capability—a critical vulnerability in VLMs.
$$
\textbf{ZSD}=\frac{1}{T-1} \sum_{t=2}^{T} \max_{1 \leq i \leq t-1}\left( p_t^{(1)} - p_t^{(i)} \right)
$$



**Recall@K and mean Average Precision** 
$$
R@K=\frac{|\mathcal{R}_q\cap\{ d_1, d_2,\ldots,d_K\} |}{|\mathcal{R}_q|}\\
    mAP= \frac{1}{Q} \sum_{i=1}^{Q} \frac{1}{m_q} \sum_{k=1}^{K} P_q(k) \delta_q(k)
$$

---

## 论文复现调整

### PromptFusion

- CLIP的获取方式：本地加载改为自动下载 `main.py line70`
  ```python
  # clip_model_path = args["file_root"] + '/' + args["backbone"] + '.pt'
    #
    # if os.path.exists(clip_model_path):
    #     clip_model, _ = clip.load(args["backbone"], device=args["device"], model_path=clip_model_path)
    # else:
    #     raise Exception("Model doesn't exist! Please manually download it!")
    # clip获取方式修改为直接下载
    clip_model, _ = clip.load("ViT-B/16", device=args["device"])
  ```

- 随机种子统一为 2 和 1993 `main.py line28`
  
  ```python
  random.seed(1993)
  np.random.seed(1993)
  torch.manual_seed(1993)
  torch.cuda.manual_seed(1993)
  torch.cuda.manual_seed_all(1993)
  ```

- `utils.py line111 imagenetr_labels`获取类名的方式
  
  ```python
  def imagenetr_labels(path):
    with open(path) as f:
        class_map = f.read().splitlines()

    classnames = []
    for map_ in class_map:
        # label, name = map_.split()
        name = map_.split("\t")[-1]
        classnames.append(name)
    return classnames
  ```

- 一些打印日志细节

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
| PromptFusion |  |  |  |  |  |

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
| PromptFusion |          |          |          |            |            |

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

## 多任务域增量

> 有两篇文章提到多任务域增量
>
> - Cross-domain Task-Agnostic Incremental Learning (X-TAIL): [LADA: scalable label-specific CLIP adapter for continual learning](http://arxiv.org/abs/2505.23271)
> - Multi-domain Task Incremental Learning (MTIL): [Preventing zero-shot transfer degradation in continual learning of vision-language models](http://arxiv.org/abs/2303.06628)

### Dataset

You can directly **download the prepared datasets** from: 👉 [https://www.modelscope.cn/datasets/ForestLuo/X-TAIL](https://www.modelscope.cn/datasets/ForestLuo/X-TAIL),
organized according to [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

Put files in the following locations and change the path in the data configure files.

```sh
Path/To/Dataset/Folder
 ├─ Aircraft
 │  ├─ images
 │  ├─ families.txt
 │  ├─ ...
 │  └─ variants.txt
 ├─ Caltech101
 │  ├─ 101_ObjectCategories
 │  └─ split_zhou_Caltech101.json
 ├─ DTD
 │  ├─ images
 │  ├─ imbd
 │  ├─ labels
 │  └─ split_zhou_DescribableTextures.json
 ├─ EuroSAT
 │  ├─ 2750
 │  └─ split_zhou_EuroSAT.json
 ├─ Flowers
 │  ├─ jpg
 │  ├─ imagelabels.mat
 │  ├─ setid.mat
 │  └─ split_zhou_OxfordFlowers.json
 ├─ Food
 │  ├─ images
 │  ├─ meta
 │  └─ split_zhou_Food101.json
 ├─ MNIST/MNIST/raw
 │  ├─ t10k-images-idx3-ubyte
 │  ├─ t10k-labels-idx1-ubyte
 │  ├─ train-images-idx3-ubyte
 │  └─ train-labels-idx1-ubyte
 ├─ Pets
 │  ├─ annotations
 │  ├─ images
 │  └─ split_zhou_OxfordPets.json
 ├─ StanfordCars
 │  ├─ cars_test
 │  ├─ cars_train
 │  ├─ devkit
 │  ├─ cars_test_annos_withlabels.mat
 │  └─ split_zhou_StanfordCars.json
 └─ Sun397
    ├─ SUN397
    ├─ ClassName.txt
    └─ split_zhou_SUN397.json
```

