from torch.cuda.amp import GradScaler, autocast# LADA: Scalable Label-Specific CLIP Adapter for Continual Learning

This is the source code for our paper "LADA: Scalable Label-Specific CLIP Adapter for Continual Learning" which has been accepted to **ICML 2025**.

## Requirements

* Python 3.10
* PyTorch 2.4.1
* Torchvision 0.19.1
* Other dependencies are listed in [requirements.txt](requirements.txt).

To install requirements, run:

```sh
conda create -n lada python=3.10 -y
conda activate lada
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Hardware

All experiments can be reproduced using a single GPU with 24GB of memory.

## Running on X-TAIL Dataset

### Prepare the Dataset

You can directly **download the prepared datasets** from: 👉 [https://www.modelscope.cn/datasets/ForestLuo/X-TAIL](https://www.modelscope.cn/datasets/ForestLuo/X-TAIL),
organized according to [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

Put files in the following locations and change the path in the data configure files [TAIL.yaml](configs/data/TAIL.yaml) and [TAIL_order2](configs/data/TAIL_order2.yaml).

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

### Reproduction

To reproduce the main result in the paper, please run

```bash
# run LADA on 16-shot order-I setting
bash scripts/run_TAIL_16shot.sh

# run LADA on 16-shot order-II setting
bash scripts/run_TAIL_16shot_order2.sh

# run LADA on full-shot order-I setting
bash scripts/run_TAIL_fullshot.sh

# run LADA on full-shot order-II setting
bash scripts/run_TAIL_fullshot_order2.sh
```

Each script will **automatically run the full experimental pipeline**, including training and evaluation, and **output the final accuracy metrics**.

### Example Output (16-shot Order I)
```bash
===========================================================================
Dataset        air.  cal.  dtd.  eur.  flo.  foo.  mni.  oxf.  sta.  sun.  
---------------------------------------------------------------------------
aircraft       48.3  75.0  36.4  37.4  64.1  83.4  43.9  87.8  65.5  61.1  
caltech101     48.8  91.6  35.8  37.2  67.2  83.9  44.0  88.0  65.4  61.4  
dtd            48.8  92.5  66.6  33.6  67.1  83.8  44.7  88.0  65.4  61.2  
eurosat        48.8  92.5  66.6  86.9  67.1  83.8  40.2  88.0  65.4  61.5  
flowers        48.8  92.7  66.8  86.9  96.3  83.8  40.2  88.0  65.4  61.5  
food101        48.8  92.7  67.8  86.9  96.4  86.1  40.2  88.0  65.4  61.5  
mnist          48.8  92.8  67.8  86.9  96.4  86.1  93.9  88.0  65.4  61.6  
oxford_pets    48.8  92.9  67.9  86.9  96.4  86.2  93.9  93.5  65.4  61.6  
stanford_cars  48.8  93.2  67.9  86.9  96.4  86.2  93.9  93.5  84.6  61.7  
sun397         49.3  93.7  69.3  86.9  96.8  86.9  93.9  93.6  84.6  76.0  

===========================================================================
EVALUATION METRICS
===========================================================================
Transfer       N/A   75.0  36.1  36.1  66.4  83.7  42.2  88.0  65.4  61.5  
Average        48.8  91.0  61.3  71.6  84.4  85.0  62.9  89.6  69.2  62.9  
Last           49.3  93.7  69.3  86.9  96.8  86.9  93.9  93.6  84.6  76.0  
===========================================================================
Transfer Mean: 61.6
Average Mean:  72.7
Last Mean:     83.1
===========================================================================
```

## Citation
If you find this repo useful for your work, please cite as:

```bibtex
@inproceedings{luo2025lada,
    title={{LADA}: Scalable Label-Specific {CLIP} Adapter for Continual Learning},
    author={Mao-Lin Luo and Zi-Hao Zhou and Tong Wei and Min-Ling Zhang},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025}
}
```

## Acknowledgment

We thank the authors for the following repositories for code reference:
[[RAIL]](https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning), [[LIFT]](https://github.com/shijxcs/LIFT/tree/main), [[CoOp]](https://github.com/KaiyangZhou/CoOp).
