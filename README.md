# 🚀 Awesome-MMCL

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" style="zoom: 60%;" /></div>
<p></p>
<div align=center><img src="https://visitor-badge.laobi.icu/badge?page_id=Awesome-MMCL&left_color=green&right_color=red" /> <img src="https://img.shields.io/github/last-commit/Geeks-Z/Awesome-MMCL" /> <img src="https://img.shields.io/github/license/Geeks-Z/Awesome-MMCL" /></div>

## 🎉 Introduction

- **Multimodal Continual Learning (MMCL)**: a resource collection for multimodal continual learning.
- **This repository collects** papers, code, datasets, and reproduced results for MMCL methods.
- [Paper reading blog](https://www.zhihu.com/column/c_1860408728035147776)

---

## 🚀 Survey

- Recent advances of multimodal continual learning: a comprehensive survey (**arXiv 2024**) [paper](https://arxiv.org/abs/2410.05352)
- When continual learning meets multimodal large language models: a survey (**arXiv 2025**) [paper](https://arxiv.org/abs/2503.01887)
- Continual learning for VLMs: a survey and taxonomy beyond forgetting (**arXiv 2025**) [paper](http://arxiv.org/abs/2508.04227)

---

## 🌟 Methods Reproduced

- `FineTune`: Baseline method which simply updates parameters on new tasks.
- `ZS-CLIP`: Baseline method which serves as a performance benchmark for the pre-trained CLIP on downstream tasks.
- `FOSTER`: Feature Boosting and Compression for Class-incremental Learning. **ECCV 2022** [[paper](https://arxiv.org/abs/2204.04662)]
- `L2P`: Learning to Prompt for Continual Learning. **CVPR 2022** [[paper](https://arxiv.org/abs/2112.08654)]
- `DualPrompt`: DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning. **ECCV 2022** [[paper](https://arxiv.org/abs/2204.04799)]
- `MEMO`: A Model or 603 Exemplars: Towards Memory-Efficient Class-Incremental Learning. **ICLR 2023 Spotlight** [[paper](https://openreview.net/forum?id=S07feAlQHgM)]
- `CODA-Prompt`: CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning. **CVPR 2023** [[paper](https://arxiv.org/abs/2211.13218)]
- `SimpleCIL`: Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need. **IJCV 2024** [[paper](https://arxiv.org/abs/2303.07338)]
- `APER`: Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need. **IJCV 2024** [[paper](https://arxiv.org/abs/2303.07338)]
- `RAPF`: Class-incremental learning with CLIP: adaptive representation adjustment and parameter fusion **ECCV 2024** [paper](https://link.springer.com/10.1007/978-3-031-72949-2_13) [code](https://github.com/linlany/RAPF) [reading notes](https://zhuanlan.zhihu.com/p/1959573948623725449)
- `TUNA`: Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning. **ICCV 2025** [[paper](https://arxiv.org/abs/2508.08165)]
- `LADA`: LADA: Scalable Label-Specific CLIP Adapter for Continual Learning  **ICML 2025** [paper](https://arxiv.org/abs/2505.23271) [code](https://github.com/MaolinLuo/LADA) [reading notes](https://zhuanlan.zhihu.com/p/1959602781133439861)
- `MG-CLIP`: Mind the gap: preserving and compensating for the modality gap in CLIP-based continual learning **ICCV 2025** [paper](https://arxiv.org/abs/2507.09118) [code](https://github.com/linlany/MindtheGap) [reading notes](https://zhuanlan.zhihu.com/p/1973063939102360605)
- `ENGINE`: External knowledge injection for CLIP-based class-incremental learning **ICCV 2025** [paper](https://arxiv.org/abs/2503.08510) [code](https://github.com/LAMDA-CL/ICCV25-ENGINE) [reading notes](https://zhuanlan.zhihu.com/p/1948401722180474817)
- `PROOF`: Learning Without Forgetting for Vision-Language Models **TPAMI 2025** [paper](https://ieeexplore.ieee.org/document/10882940/) [code](https://github.com/LAMDA-CL/PROOF) [reading notes](https://zhuanlan.zhihu.com/p/1959573532569768109)
- `CLG-CBM`: Language Guided Concept Bottleneck Models for Interpretable Continual Learning. **CVPR 2025** [[paper](https://arxiv.org/abs/2503.23283)]
- `BOFA`: BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning. **AAAI 2026** [[paper](https://arxiv.org/abs/2511.11421)]
- `AREA`: Attribute Extraction and Aggregation for CLIP-Based Class-Incremental Learning. **ICML 2026** [[paper](https://arxiv.org/abs/2605.28809)] [[code](https://github.com/LAMDA-CL/ICML2026-AREA)]
- `MoE-Adapters`: Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters. **CVPR 2024** [[paper](https://arxiv.org/abs/2403.11549)] [[code](https://github.com/JiazuoYu/MoE-Adapters4CL)]
- `PromptFusion`: PromptFusion: Decoupling Stability and Plasticity for Continual Learning. **ECCV 2024** [[paper](https://arxiv.org/abs/2303.07223)] [[code](https://github.com/HaoranChen/PromptFusion)]

### TODO

- `LADA`: LADA: Scalable Label-Specific CLIP Adapter for Continual Learning **ICML 2025** [paper](https://arxiv.org/abs/2505.23271) [code](https://github.com/MaolinLuo/LADA) [reading notes](https://zhuanlan.zhihu.com/p/1959602781133439861)

---

## ☄️ How to Use

### 🕹️ Clone

Clone this GitHub repository:

```
git clone https://gitee.com/geeks_z/Awesome-MMCIL.git
cd Awesome-MMCIL
```

### 🗂️ Dependencies

1. [torch 2.0.1+cu118](https://github.com/pytorch/pytorch)
2. [torchvision 0.15.2+cu118](https://github.com/pytorch/vision)
3. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models)
4. [tqdm 4.66.2](https://github.com/tqdm/tqdm)
5. [numpy 1.26.3](https://github.com/numpy/numpy)
6. [scipy 1.12.0](https://github.com/scipy/scipy)
7. [easydict 1.13](https://github.com/makinacorpus/easydict)
8. [open-clip 2.30.0](https://github.com/mlfoundations/open_clip/releases/tag/v2.30.0)

### 🔑 Run experiment

The repository has a native MMCL framework and standalone reproductions. Use the matching entry point for the method you want to run.

#### Native MMCL framework

`FineTune`, `ZS-CLIP`, `FOSTER`, `L2P`, `DualPrompt`, `MEMO`, `CODA-Prompt`, `SimpleCIL`, `APER`, `RAPF`, `TUNA`, `CLG-CBM`, `ENGINE`, `PROOF`, `BOFA`, and `AREA` use the root-level `main.py`, `configs/`, `models/`, and `scripts/` layout. Edit the target JSON configuration, then run a single configuration with:

   ```bash
   python main.py --config=./configs/[METHOD]/[CONFIG].json
   ```

For a method's configured sequence, use its corresponding `scripts/run_[method].sh` launcher.

`hyper-parameters`

   - **model_name**: Select one of the implemented methods: `finetune`, `zs_clip`, `foster`, `memo`, `simplecil`, `l2p`, `dualprompt`, `coda`, `aper`, `tuna`, `rapf`, `CLG-CBM`, `mg_clip`, `proof`, `engine`, `bofa` or `area`.
   - **init_cls**: The number of classes in the initial incremental stage. As the configuration of CIL includes different settings with varying class numbers at the outset, our framework accommodates diverse options for defining the initial stage.
   - **increment**: The number of classes in each incremental stage $i$, $i$ > 1. By default, the number of classes is equal across all incremental stages.
   - **backbone_type**: The backbone network of the incremental model. It can be selected from a variety of pre-trained models available in the Timm library, such as **LAION-400M** and **OpenAI**,  for
     the CLIP with **ViT-B/16**.
   - **seed**: The random seed is utilized for shuffling the class order. It is set to 1993 by default, following the benchmark setting iCaRL.
   - **fixed_memory**: a Boolean parameter. When set to true, the model will maintain a fixed amount of memory per class. Alternatively, when set to false, the model will preserve dynamic memory allocation per class.
   - **memory_size**: The total number of exemplars in the incremental learning process. If `fixed_memory` is set to false, assuming there are $K$ classes at the current stage, the model will preserve $\left[\frac{{memory-size}}{K}\right]$ exemplars for each class. **ZS-CLIP, SimpleCIL, TUNA, CLG-CBM, MG-CLIP, ENGINE, BOFA and AREA do not require exemplars.** Therefore, parameters related to exemplars are not used.
   - **memory_per_class**: If `fixed memory` is set to true, the model will preserve a fixed number of `memory_per_class` exemplars for each class.

#### Standalone reproductions

`PromptFusion` and `MoE-Adapters` keep their upstream project layouts under `PromptFusion-main/` and `MoE-Adapters4CL-MoE-Adapters/cil/`. Use the root launchers, which set the shared dataset root and select one GPU:

```bash
GPU_ID=5 bash scripts/run_promptfusion.sh
GPU_ID=6 bash scripts/run_moe_adapters.sh
```

`LADA` is also standalone, but it reproduces the X-TAIL multi-domain protocol rather than the native CIL protocol. Its entry point is `LADA-main/run_mmcl_seeds.sh`; it requires the X-TAIL dataset and the upstream LADA environment specified in `LADA-main/README.md` (PyTorch 2.4.1), not the root `main.py` runner.

## 📊 Reproduced Results

Results parsed from the training logs are maintained in [MMCL_OpenCLIP_ViT-B16_LAION-400M_Baselines.xlsx](MMCL_OpenCLIP_ViT-B16_LAION-400M_Baselines.xlsx) and [MMCL_OpenAI_CLIP_ViT-B16_Baselines.xlsx](MMCL_OpenAI_CLIP_ViT-B16_Baselines.xlsx). The workbook contains one sheet for each seed (`0`, `42`, `1993`, and `2026`), the benchmark comparison sheet, and the current experiment-progress sheet. Metric definitions and notes are available in [MMCL_Baselines.md](MMCL_Baselines.md) and [MMCL_Baselines.en.md](MMCL_Baselines.en.md).

- **Datasets**

  | Dataset       | training instances | testing instances | Classes | Drive                                                        |
  | ------------- | ------------------ | ----------------- | ------- | ------------------------------------------------------------ |
  | CIFAR100      | 50,000             | 10,000            | 100     | will be automatically downloaded by the code.                |
  | CUB200-2011   | 9,430              | 2,358             | 200     | Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or OneDrive [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc) |
  | ImageNet-R    | 24,000             | 6,000             | 200     | Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW) |
  | ImageNet-A    | 5,981              | 1,519             | 200     |                                                              |
  | ObjectNet     | 26,509             | 6,628             | 200     | Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EZFv9uaaO1hBj7Y40KoCvYkBnuUZHnHnjMda6obiDpiIWw?e=4n8Kpy) You can also refer to the [filelist](https://drive.google.com/file/d/147Mta-HcENF6IhZ8dvPnZ93Romcie7T6/view?usp=sharing) and processing [code](https://github.com/zhoudw-zdw/RevisitingCIL/issues/2#issuecomment-2280462493) if the file is too large to download. |
  | Omnibenchmark | 89,697             | 5,983             | 300     |                                                              |
  | VTAB          | 1,796              | 8,619             | 50      |                                                              |
  | Cars          | 4,135              | 4,083             | 100     | Google Drive: [link](https://drive.google.com/file/d/1D8ReAuOPenWi6SMNUrOZhbm6ViyhDHbL/view?usp=sharing) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EbT1XAstg51Mpy82uHM0D2EBJLrtzmr_V64jeBRjqyyTnQ?e=h6g1rM) |
  | UCF           | 10,553             | 2,639             | 100     | Google Drive: [link](https://drive.google.com/file/d/1Ng4w310_VDqpKbc7eYaumXTOiDxI02Wc/view?usp=sharing) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EU2qHQXjASdLh1jIl6ihZmcB6G2KvqmSw-sTlZKDE6xPbg?e=7ezvTr) |
  | Aircraft      | 6,667              | 3,333             | 100     | Google Drive: [link](https://drive.google.com/file/d/1xI5r1fU0d6Nff51HuOo5w-e4sGEP46Z2/view?usp=drive_link) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/ETVliZnmPY9AvZZgcFFJ6jMB2c7TRvcq7-gso2Aqvdl_VQ?e=pWXqdP) |
  | Food          | 79,988             | 20,012            | 100     | Google Drive: [link](https://drive.google.com/file/d/1rupzXpwrbxki4l-RVmsRawhz1Cm0lDY5/view?usp=drive_link) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/Eb4xfptD4L5Egus-SiYxrIcBDH1VewLGp4kzyACGF_Na_w?e=duA3Ia) |
  | SUN           | 72,870             | 18,179            | 300     | OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EcQq1-1pFulKstYtdknB4O8BGo0hnlDRarAwB4wFEgkx0Q?e=YZ0xYV) |
  | TV100         | 41,716             | 4,000             | 100     | OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/r/personal/ky2409911_365_nju_edu_cn/Documents/TV100/TV100.zip?csf=1&web=1&e=XNpitj) |

  The instance and class counts above refer to the preprocessed benchmark subsets used by this framework, which may differ from the complete original datasets.

  When training **not** on `CIFAR100`, you should specify the folder of your dataset in `utils/data.py`.

  ```python
  def download_data(self):
          assert 0,"You should specify the folder of your dataset"
          train_dir = '[DATA-PATH]/train/'
          test_dir = '[DATA-PATH]/val/'
  ```

- ### Dataset split

  - B-$m$ Inc-$n$: $m$ denotes the number of classes in the initial incremental stage, and $n$ denotes the number of classes in each subsequent incremental stage.

- ### Backbone

  | Model                  | Training Data                                        | Publisher                 | Characteristics                                              | Model Scale                        | Creation Method                                              |
  | ---------------------- | ---------------------------------------------------- | ------------------------- | ------------------------------------------------------------ | ---------------------------------- | ------------------------------------------------------------ |
  | **OpenAI CLIP**        | 400M internal image-text pairs collected by OpenAI   | OpenAI                    | Official version with stable performance, but training data is not public | ViT-B/32, ViT-B/16, ViT-L/14, etc. | `model, preprocess = clip.load("ViT-B/32", device=device)`   |
  | **OpenCLIP LAION400M** | Public LAION-400M dataset with 400M image-text pairs | LAION open-source project | Fully open source; training data is public and reproducible  | Multiple architectures and scales  | `model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')` |

---

## 🤗 Multi-domain Incremental Learning

> Two papers mention multi-domain incremental learning:
>
> - Cross-domain Task-Agnostic Incremental Learning (X-TAIL): [LADA: scalable label-specific CLIP adapter for continual learning](http://arxiv.org/abs/2505.23271)
> - Multi-domain Task Incremental Learning (MTIL): [Preventing zero-shot transfer degradation in continual learning of vision-language models](http://arxiv.org/abs/2303.06628)

**Dataset**

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

## 🎈 Acknowledgments

- [C3Box](https://github.com/LAMDA-CL/C3Box)
- [PyCIL](https://github.com/G-U-N/PyCIL)
- [PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT)
- [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
