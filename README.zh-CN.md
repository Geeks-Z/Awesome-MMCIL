# 🚀 Awesome-MMCL

[English](README.md) | [简体中文](README.zh-CN.md)

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" style="zoom: 60%;" /></div>
<p></p>
<div align=center><img src="https://visitor-badge.laobi.icu/badge?page_id=Awesome-MMCL&left_color=green&right_color=red" /> <img src="https://img.shields.io/github/last-commit/Geeks-Z/Awesome-MMCL" /> <img src="https://img.shields.io/github/license/Geeks-Z/Awesome-MMCL" /></div>

## 🎉 简介

- **多模态增量学习**：Multimodal Continual Learning (MMCL)
- **汇总**多模态增量学习的资源、代码和论文，并对部分方法进行复现
- [论文阅读博客](https://www.zhihu.com/column/c_1860408728035147776)

---

## 🚀 综述

- Recent advances of multimodal continual learning: a comprehensive survey (**arXiv 2024**) [paper](https://arxiv.org/abs/2410.05352)
- When continual learning meets multimodal large language models: a survey (**arXiv 2025**) [paper](https://arxiv.org/abs/2503.01887)
- Continual learning for VLMs: a survey and taxonomy beyond forgetting (**arXiv 2025**) [paper](http://arxiv.org/abs/2508.04227)

---

## 🌟 已复现方法

- `FineTune`: 基线方法，直接在新任务上更新参数。
- `ZS-CLIP`: 基线方法，用于评估预训练 CLIP 在下游任务上的性能。
- `FOSTER`: Feature Boosting and Compression for Class-incremental Learning. **ECCV 2022** [[paper](https://arxiv.org/abs/2204.04662)]
- `L2P`: Learning to Prompt for Continual Learning. **CVPR 2022** [[paper](https://arxiv.org/abs/2112.08654)]
- `DualPrompt`: DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning. **ECCV 2022** [[paper](https://arxiv.org/abs/2204.04799)]
- `MEMO`: A Model or 603 Exemplars: Towards Memory-Efficient Class-Incremental Learning. **ICLR 2023 Spotlight** [[paper](https://openreview.net/forum?id=S07feAlQHgM)]
- `CODA-Prompt`: CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning. **CVPR 2023** [[paper](https://arxiv.org/abs/2211.13218)]
- `SimpleCIL`: Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need. **IJCV 2024** [[paper](https://arxiv.org/abs/2303.07338)]
- `APER`: Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need. **IJCV 2024** [[paper](https://arxiv.org/abs/2303.07338)]
- `RAPF`: Class-incremental learning with CLIP: adaptive representation adjustment and parameter fusion **ECCV 2024** [paper](https://link.springer.com/10.1007/978-3-031-72949-2_13) [code](https://github.com/linlany/RAPF) [论文阅读](https://zhuanlan.zhihu.com/p/1959573948623725449)
- `MoE-Adapters`: Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters **CVPR 2024** [paper](https://arxiv.org/abs/2403.11549) [code](https://github.com/JiazuoYu/MoE-Adapters4CL) [论文阅读](https://zhuanlan.zhihu.com/p/1915756018090119546)
- `PromptFusion`: PromptFusion: Decoupling Stability and Plasticity for Continual Learning **ECCV 2024** [paper](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/2303.07223) [code](https://github.com/HaoranChen/PromptFusion) [论文阅读](https://zhuanlan.zhihu.com/p/1977048983483478476)
- `TUNA`: Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning. **ICCV 2025** [[paper](https://arxiv.org/abs/2508.08165)]
- `LADA`: LADA: Scalable Label-Specific CLIP Adapter for Continual Learning  **ICML 2025** [paper](https://arxiv.org/abs/2505.23271) [code](https://github.com/MaolinLuo/LADA) [论文阅读](https://zhuanlan.zhihu.com/p/1959602781133439861)
- `MG-CLIP`: Mind the gap: preserving and compensating for the modality gap in CLIP-based continual learning **ICCV 2025** [paper](https://arxiv.org/abs/2507.09118) [code](https://github.com/linlany/MindtheGap) [论文阅读](https://zhuanlan.zhihu.com/p/1973063939102360605)
- `ENGINE`: External knowledge injection for CLIP-based class-incremental learning **ICCV 2025** [paper](https://arxiv.org/abs/2503.08510) [code](https://github.com/LAMDA-CL/ICCV25-ENGINE) [论文阅读](https://zhuanlan.zhihu.com/p/1948401722180474817)
- `PROOF`: Learning Without Forgetting for Vision-Language Models **TPAMI 2025** [paper](https://ieeexplore.ieee.org/document/10882940/) [code](https://github.com/LAMDA-CL/PROOF) [论文阅读](https://zhuanlan.zhihu.com/p/1959573532569768109)
- `CLG-CBM`: Language Guided Concept Bottleneck Models for Interpretable Continual Learning. **CVPR 2025** [[paper](https://arxiv.org/abs/2503.23283)]
- `BOFA`: BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning. **AAAI 2026** [[paper](https://arxiv.org/abs/2511.11421)]

---

## ☄️ 使用方法

### 🕹️ 克隆

克隆本 GitHub 仓库：

```
git clone https://gitee.com/geeks_z/Awesome-MMCIL.git
cd Awesome-MMCIL
```

### 🗂️ 依赖

1. [torch 2.0.1](https://github.com/pytorch/pytorch)
2. [torchvision 0.15.2](https://github.com/pytorch/vision)
3. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models)
4. [tqdm](https://github.com/tqdm/tqdm)
5. [numpy](https://github.com/numpy/numpy)
6. [scipy](https://github.com/scipy/scipy)
7. [easydict](https://github.com/makinacorpus/easydict)
8. [open-clip 2.17.1](https://github.com/mlfoundations/open_clip/releases/tag/v2.17.1)

### 🔑 运行实验

1. 编辑 `[MODEL NAME].json` 文件，设置全局配置和超参数。

2. 运行：

   ```bash
   python main.py --config=./configs/[METHOD]/[CONFIG].json
   ```

3. `hyper-parameters`

   - **model_name**: 模型名称应从已实现的方法中选择：`finetune`、`zs_clip`、`foster`、`memo`、`simplecil`、`l2p`、`dualprompt`、`coda`、`aper`、`tuna`、`rapf`、`CLG-CBM`、`mg_clip`、`proof`、`engine` 或 `bofa`。
   - **init_cls**: 初始增量阶段的类别数量。CIL 配置通常包含不同的初始类别数设置，本框架支持多种初始阶段定义方式。
   - **increment**: 第 $i$ 个增量阶段的类别数量，其中 $i$ > 1。默认情况下，所有增量阶段的类别数量相同。
   - **backbone_type**: 增量模型的骨干网络。可从 Timm 库提供的多种预训练模型中选择，例如基于 **ViT-B/16** 的 **LAION-400M** 和 **OpenAI** CLIP。
   - **seed**: 用于打乱类别顺序的随机种子。默认设置为 1993，沿用 iCaRL 的 benchmark 设置。
   - **fixed_memory**: 布尔参数。设为 true 时，每类保留固定数量的 memory；设为 false 时，每类 memory 数量动态分配。
   - **memory_size**: 增量学习过程中的 exemplar 总数。如果 `fixed_memory` 为 false，假设当前阶段有 $K$ 个类别，模型将为每类保留 $\left[\frac{{memory-size}}{K}\right]$ 个 exemplar。**ZS-CLIP、SimpleCIL、TUNA、CLG-CBM、MG-CLIP、ENGINE 和 BOFA 不需要 exemplar**，因此不会使用 exemplar 相关参数。
   - **memory_per_class**: 如果 `fixed memory` 为 true，模型将为每个类别保留固定数量的 `memory_per_class` exemplars。

## 📊 实验结果

训练日志解析后的结果维护在 [MMCL实验结果.xlsx](logs/MMCL实验结果.xlsx) 中。工作簿包含四个 seed sheet（`0`、`42`、`1993`、`2026`）以及一个 benchmark 对比 sheet。指标定义和复现实验说明见 [实验结果.md](实验结果.md) 与 [实验结果.en.md](实验结果.en.md)。

## 📚 数据集

| 数据集        | 训练样本数         | 测试样本数        | 类别数  | 链接                                                         | 简介                                     | 下载地址                                                     |
| ------------- | ------------------ | ----------------- | ------- | ------------------------------------------------------------ | ---------------------------------------- | ------------------------------------------------------------ |
| CIFAR100      | 50,000             | 10,000            | 100     |                                                              |                                          | 代码会自动下载。                                             |
| CUB200-2011   | 9,430              | 2,358             | 200     |                                                              | 加州理工学院2010年提出的鸟类细粒度数据集 | Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or OneDrive [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc) |
| ImageNet-R    | 24,000             | 6,000             | 200     |                                                              |                                          | Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW) |
| ImageNet-A    | 5,981              | 1,519             | 200     |                                                              |                                          |                                                              |
| ObjectNet     | 26,509             | 6,628             | 200     | [https:objectnet.dev/download.html](https:objectnet.dev/download.html) |                                          | Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EZFv9uaaO1hBj7Y40KoCvYkBnuUZHnHnjMda6obiDpiIWw?e=4n8Kpy) You can also refer to the [filelist](https://drive.google.com/file/d/147Mta-HcENF6IhZ8dvPnZ93Romcie7T6/view?usp=sharing) and processing [code](https://github.com/zhoudw-zdw/RevisitingCIL/issues/2#issuecomment-2280462493) if the file is too large to download. |
| Omnibenchmark | 89,697             | 5,983             | 300     |                                                              |                                          |                                                              |
| VTAB          | 1,796              | 8,619             | 50      |                                                              |                                          |                                                              |
| Cars          |                    |                   |         |                                                              |                                          | Google Drive: [link](https://drive.google.com/file/d/1D8ReAuOPenWi6SMNUrOZhbm6ViyhDHbL/view?usp=sharing  ) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EbT1XAstg51Mpy82uHM0D2EBJLrtzmr_V64jeBRjqyyTnQ?e=h6g1rM) |
| UCF           |                    |                   |         |                                                              |                                          | Google Drive: [link](https://drive.google.com/file/d/1Ng4w310_VDqpKbc7eYaumXTOiDxI02Wc/view?usp=sharing) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EU2qHQXjASdLh1jIl6ihZmcB6G2KvqmSw-sTlZKDE6xPbg?e=7ezvTr) |
| Aircraft      |                    |                   |         |                                                              |                                          | Google Drive: [link](https://drive.google.com/file/d/1xI5r1fU0d6Nff51HuOo5w-e4sGEP46Z2/view?usp=drive_link) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/ETVliZnmPY9AvZZgcFFJ6jMB2c7TRvcq7-gso2Aqvdl_VQ?e=pWXqdP) |
| Food          |                    |                   |         |                                                              |                                          | Google Drive: [link](https://drive.google.com/file/d/1rupzXpwrbxki4l-RVmsRawhz1Cm0lDY5/view?usp=drive_link) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/Eb4xfptD4L5Egus-SiYxrIcBDH1VewLGp4kzyACGF_Na_w?e=duA3Ia) |
| SUN           |                    |                   |         |                                                              |                                          | OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EcQq1-1pFulKstYtdknB4O8BGo0hnlDRarAwB4wFEgkx0Q?e=YZ0xYV) |
| TV100         |                    |                   |         |                                                              |                                          | OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/r/personal/ky2409911_365_nju_edu_cn/Documents/TV100/TV100.zip?csf=1&web=1&e=XNpitj) |

如果训练数据集**不是** `CIFAR100`，需要在 `utils/data.py` 中指定数据集路径。

```python
    def download_data(self):
        assert 0,"You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'
```

---



## 🤗 多任务域增量

> 有两篇文章提到多任务域增量
>
> - Cross-domain Task-Agnostic Incremental Learning (X-TAIL): [LADA: scalable label-specific CLIP adapter for continual learning](http://arxiv.org/abs/2505.23271)
> - Multi-domain Task Incremental Learning (MTIL): [Preventing zero-shot transfer degradation in continual learning of vision-language models](http://arxiv.org/abs/2303.06628)

**数据集**

可以直接从 👉 [https://www.modelscope.cn/datasets/ForestLuo/X-TAIL](https://www.modelscope.cn/datasets/ForestLuo/X-TAIL) **下载整理好的数据集**，
数据组织方式遵循 [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)。

将文件放置在以下位置，并在数据配置文件中修改对应路径。

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

## 🎈 致谢

- [C3Box](https://github.com/LAMDA-CL/C3Box)
- [PyCIL](https://github.com/G-U-N/PyCIL)
- [PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT)
- [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
