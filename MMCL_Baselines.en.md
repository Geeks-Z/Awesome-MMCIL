# 📊 Results and Evaluation Metrics

[中文](实验结果.md) | [English](实验结果.en.md)

> - [Continual learning for VLMs: a survey and taxonomy beyond forgetting](http://arxiv.org/abs/2508.04227)
> - [Recent advances of multimodal continual learning: a comprehensive survey](http://arxiv.org/abs/2410.05352)


<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20251203104009.png" style="zoom: 60%;" /></div>

> Let $p^j_i$ denote the model performance or accuracy on task $j$ after finishing training on task $i$, where $i=0$ denotes the pre-trained state.



**Average Accuracy** remains fundamental, measuring mean task performance over all learning stages to gauge overall proficiency.

> For conventional continual learning, only scores above the diagonal are meaningful, since they cannot give zero-shot predictions on unseen tasks. $\bar{\mathcal{A}} = \frac{1}{T}\sum_{i=1}^{T}( \frac{1}{i} \sum_{j=1}^{i}p_i^j)$

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
> This measures the difference between the model's performance on a previous task $i$ after learning the final task $T$ and its performance immediately after learning task $i$. The value is usually negative, shown as the blue region in the figure, and reflects retention ability.
>
> - A large drop means the model forgets much of the knowledge learned earlier.
> - A small drop means the model retains most of the earlier knowledge.
> - A positive value means later learning improves performance on earlier tasks.

$$
\textbf{BWT}=\frac{1}{T-1} \sum_{t=1}^{T-1} \left(p_t^{(T)} -p_t^{(t)}  \right)
$$

**Forward Transfer(FWT)** 

> <div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20260127095811.png" style="zoom: 80%;" /></div>
>
> Taking the final task $T$ as an example, this metric reflects how learning the previous $T-1$ tasks affects performance on task $T$ before task $T$ itself is learned. It subtracts the model's initial performance on task $T$ from its performance on task $T$ after learning the first $T-1$ tasks, shown as the green region in the figure. A positive value indicates positive transfer from previous tasks and reflects **transfer ability**.
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

# 📝 Reproduction Notes

## Notes on PROOF and ENGINE

1. PROOF uses memory by design, so memory is also added when reproducing methods such as L2P, DualPrompt, and CODA-Prompt.

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20260301102903.png" style="zoom: 80%;" /></div>

2. ENGINE uses no memory for any method.

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20260301102957.png" style="zoom: 80%;" /></div>

---

## 🛠️ C3Box Code Changes

1. Comment out memory settings in the configuration files for L2P, DualPrompt, and CODA-Prompt.

2. Comment out the memory call in `models/l2p.py`.

   ```python
           if len(self._multiple_gpus) > 1:
               print('Multiple GPUs')
               self._network = nn.DataParallel(self._network, self._multiple_gpus)
           self._train(self.train_loader, self.test_loader)
           if len(self._multiple_gpus) > 1:
               self._network = self._network.module
           # if self.args["memory_size"] > 0:
           #     self.build_rehearsal_memory(data_manager, self.samples_per_class)
   ```

3. In the `_init_train` method of `models/l2p.py`, uncomment `logits[:, :self._known_classes] = float('-inf')` to reduce changes to old-class classifier weights.

   ```python
   def _init_train(self, train_loader, test_loader, optimizer, scheduler):
           prog_bar = tqdm(range(self.args['tuned_epoch']))
           for _, epoch in enumerate(prog_bar):
               self._network.backbone.train()
               self._network.original_backbone.eval()
   
               losses = 0.0
               correct, total = 0, 0
               for i, (_, inputs, targets) in enumerate(train_loader):
                   inputs, targets = inputs.to(self._device), targets.to(self._device)
   
                   output = self._network(inputs, task_id=self._cur_task, train=True)
                   logits = output["logits"][:, :self._total_classes]
                   logits[:, :self._known_classes] = float('-inf') 
   ```

---

## 🔎 PromptFusion

- CLIP loading: switch from local loading to automatic downloading at `main.py line70`.
  ```python
  # clip_model_path = args["file_root"] + '/' + args["backbone"] + '.pt'
    #
    # if os.path.exists(clip_model_path):
    #     clip_model, _ = clip.load(args["backbone"], device=args["device"], model_path=clip_model_path)
    # else:
    #     raise Exception("Model doesn't exist! Please manually download it!")
    # Change CLIP loading to direct download
    clip_model, _ = clip.load("ViT-B/16", device=args["device"])
  ```

- Use unified random seeds 2 and 1993 at `main.py line28`.
  
  ```python
  random.seed(1993)
  np.random.seed(1993)
  torch.manual_seed(1993)
  torch.cuda.manual_seed(1993)
  torch.cuda.manual_seed_all(1993)
  ```

- How `imagenetr_labels` obtains class names at `utils.py line111`.
  
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

- Some logging details.

---

# 📊 Current Reproduced Results

The log-derived results are maintained in [MMCL_Baselines.xlsx](MMCL_Baselines.xlsx). The workbook has one sheet for each seed: `Seed0`, `Seed42`, `Seed1993`, and `Seed2026`. The tables below preserve the reproduction context and historical comparisons; when a value differs, the workbook and its corresponding log are the source of truth.

## 👨‍🏫 Experimental Details

- ### Dataset split

  - B-$m$ Inc-$n$: $m$ denotes the number of classes in the initial incremental stage, and $n$ denotes the number of classes in each subsequent incremental stage.
  
- ### Backbone

  | Model                  | Training Data                          | Publisher          | Characteristics                      | Model Scale                    | Creation Method                                              |
  | ---------------------- | -------------------------------------- | ------------------ | ------------------------------------ | ------------------------------ | ------------------------------------------------------------ |
  | **OpenAI CLIP**        | 400M internal image-text pairs collected by OpenAI | OpenAI             | Official version with stable performance, but training data is not public | ViT-B/32, ViT-B/16, ViT-L/14, etc. | `model, preprocess = clip.load("ViT-B/32", device=device)`   |
  | **OpenCLIP LAION400M** | Public LAION-400M dataset with 400M image-text pairs | LAION open-source project | Fully open source; training data is public and reproducible | Multiple architectures and scales | `model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')` |

  > All experiments use `OpenAI_CLIP`.

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

## Seed1993 Overview

The following tables preserve the seed=1993 overview for the three dataset groups. `A_bar` denotes Average Accuracy and `A_B` denotes the final-stage Last Accuracy. The workbook contains the complete method and dataset matrix.

**Table 1: Aircraft, CIFAR100, Cars**

| Method | Air B0 A_bar | Air B0 A_B | Air B50 A_bar | Air B50 A_B | CIF B0 A_bar | CIF B0 A_B | CIF B50 A_bar | CIF B50 A_B | Car B0 A_bar | Car B0 A_B | Car B50 A_bar | Car B50 A_B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleCIL | 59.06 | 47.94 | 52.86 | 47.94 | 84.15 | 76.63 | 80.20 | 76.63 | 92.11 | 86.97 | 89.05 | 86.97 |
| ZS-CLIP | 26.61 | 17.16 | 21.66 | 17.16 | 81.81 | 71.38 | 76.49 | 71.38 | 82.90 | 76.73 | 78.74 | 76.73 |
| BOFA | 70.85 | 60.37 | 66.31 | 61.48 | 86.05 | 79.18 | 83.07 | 79.45 | 94.26 | 90.18 | 92.05 | 90.47 |
| CLG-CBM | 65.76 | 55.33 | 59.61 | 55.36 | - | - | - | - | - | - | - | - |
| AREA | 66.01 | 53.65 | 67.50 | 63.52 | 88.52 | 82.69 | 85.96 | 83.22 | 94.34 | 90.35 | 92.30 | 90.77 |

**Table 2: ImageNet-R, CUB, UCF**

| Method | INR B0 A_bar | INR B0 A_B | INR B100 A_bar | INR B100 A_B | CUB B0 A_bar | CUB B0 A_B | CUB B100 A_bar | CUB B100 A_B | UCF B0 A_bar | UCF B0 A_B | UCF B50 A_bar | UCF B50 A_B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleCIL | 81.13 | 74.55 | 76.92 | 74.55 | 83.83 | 77.52 | 79.77 | 77.52 | 90.41 | 85.79 | 88.08 | 85.79 |
| ZS-CLIP | 83.50 | 77.32 | 79.72 | 77.32 | 74.21 | 63.06 | 67.84 | 63.06 | 75.88 | 67.79 | 71.68 | 67.79 |
| BOFA | 85.12 | 79.40 | 81.94 | 79.97 | 86.59 | 80.53 | 83.09 | 80.83 | 93.17 | 88.82 | 92.60 | 89.50 |
| CLG-CBM | 84.49 | 77.77 | 81.32 | 78.00 | - | - | - | - | 95.09 | 91.66 | - | - |
| AREA | - | - | 82.83 | 80.45 | - | - | 83.99 | 80.87 | - | - | - | - |

**Table 3: SUN, Food, ObjectNet**

| Method | SUN B0 A_bar | SUN B0 A_B | SUN B150 A_bar | SUN B150 A_B | Food B0 A_bar | Food B0 A_B | Food B50 A_bar | Food B50 A_B | Obj B0 A_bar | Obj B0 A_B | Obj B100 A_bar | Obj B100 A_B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleCIL | 82.15 | 75.60 | 78.65 | 75.60 | 87.89 | 81.70 | 84.78 | 81.70 | 52.06 | 40.13 | 45.11 | 40.13 |
| ZS-CLIP | 79.45 | 72.14 | 74.98 | 72.14 | 87.87 | 81.99 | 84.78 | 81.99 | 38.43 | 26.43 | 31.12 | 26.43 |
| BOFA | 84.24 | 77.75 | - | - | 88.60 | 82.63 | - | - | 59.26 | 47.18 | 52.31 | 47.33 |
| CLG-CBM | 84.63 | 77.94 | - | - | - | - | - | - | 58.17 | 44.89 | 49.72 | 43.39 |

## 📣 Historical Reproduction (C3Box) VS ENGINE

> This section records an earlier comparison between the C3Box reproduction and the results reported by ENGINE. It is not a substitute for the complete four-seed workbook. Positive values mean the reproduced result is higher, and negative values mean it is lower.

**Table 1: Aircraft, CIFAR100, Cars**

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20260228154900.png" style="zoom: 80%;" /></div>

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20260228154908.png" style="zoom: 80%;" /></div>

| Method      | Air B0 $\bar{A}$ | Air B0 $A_B$ | Air B50 $\bar{A}$ | Air B50 $A_B$ | CIF B0 $\bar{A}$ | CIF B0 $A_B$ | CIF B50 $\bar{A}$ | CIF B50 $A_B$ | Car B0 $\bar{A}$ | Car B0 $A_B$ | Car B50 $\bar{A}$ | Car B50 $A_B$ |
| ----------- | ---------------- | ------------ | ----------------- | ------------- | ---------------- | ------------ | ----------------- | ------------- | ---------------- | ------------ | ----------------- | ------------- |
| Finetune    | +12.78           | +7.83        | +3.38             | +3.45         | +12.97           | +5.01        | +18.57            | +6.46         | +2.22            | +2.79        | +1.61             | +1.29         |
| SimpleCIL   | -0.18            | -0.15        | -0.19             | -0.15         | 0                | 0            | 0                 | 0             | +0.07            | +0.12        | +0.09             | +0.12         |
| ZS-CLIP     | -0.05            | -0.06        | -0.04             | -0.06         | 0                | 0            | 0                 | 0             | +0.30            | +0.36        | +0.42             | +0.36         |
| L2P         | +1.59            | +1.83        | +6.69             | +2.19         | -0.32            | -0.56        | -1.49             | -3.34         | +7.08            | +8.74        | +1.39             | -2.30         |
| DualPrompt  | +0.93            | +2.52        | -0.07             | -0.12         | +1.95            | +3.17        | -0.99             | -3.27         | +5.91            | +8.01        | -4.82             | -9.11         |
| CODA-Prompt | **-5.90**        | **-5.07**    | -1.63             | -2.73         | -1.73            | -4.88        | **-7.77**         | **-14.79**    | -2.56            | -1.89        | **-9.22**         | **-16.60**    |
| RAPF        | **-6.76**        | +0.21        | -2.31             | +0.90         | +1.33            | +2.61        | +2.43             | +3.30         | -1.58            | +5.21        | +1.63             | +3.84         |
| ENGINE      | +0.05            | -0.18        | +0.07             | -0.06         | -0.04            | +0.07        | 0                 | +0.02         | +0.02            | +0.05        | +0.03             | -0.05         |

---

**Table 2: ImageNet-R, CUB, UCF**

| Method      | INR B0 $\bar{A}$ | INR B0 $A_B$ | INR B100 $\bar{A}$ | INR B100 $A_B$ | CUB B0 $\bar{A}$ | CUB B0 $A_B$ | CUB B100 $\bar{A}$ | CUB B100 $A_B$ | UCF B0 $\bar{A}$ | UCF B0 $A_B$ | UCF B50 $\bar{A}$ | UCF B50 $A_B$ |
| ----------- | ---------------- | ------------ | ------------------ | -------------- | ---------------- | ------------ | ------------------ | -------------- | ---------------- | ------------ | ----------------- | ------------- |
| Finetune    | +22.21           | +7.40        | +19.36             | +6.54          | +24.06           | +7.08        | +20.41             | +7.97          | +14.18           | +4.59        | +7.37             | +3.82         |
| SimpleCIL   | +0.07            | +0.07        | +0.08              | +0.07          | +0.01            | 0            | +0.02              | 0              | -0.03            | +0.11        | -0.04             | +0.11         |
| ZS-CLIP     | +0.13            | +0.15        | +0.15              | +0.15          | -0.17            | 0            | -0.12              | 0              | +0.38            | +0.15        | +0.24             | +0.15         |
| L2P         | +5.51            | +5.76        | +1.17              | -3.69          | +1.25            | -0.25        | **-8.04**          | **-15.06**     | +3.32            | +4.89        | -1.24             | **-6.86**     |
| DualPrompt  | +5.40            | +7.25        | -1.34              | **-5.68**      | +4.28            | +5.52        | -4.47              | **-7.33**      | +1.75            | +3.57        | -2.90             | **-6.82**     |
| CODA-Prompt | +3.10            | +2.05        | **-7.08**          | **-15.33**     | -2.32            | **-6.02**    | **-13.95**         | **-20.90**     | -4.01            | **-5.45**    | **-10.88**        | **-19.93**    |
| RAPF        | +2.37            | +5.15        | +3.44              | +5.94          | **-5.38**        | -2.93        | **-5.34**          | **-6.23**      | -1.75            | +1.52        | -2.51             | -2.01         |
| ENGINE      | +0.05            | +0.43        | -0.01              | -0.11          | -0.05            | +0.33        | +0.52              | +0.68          | +0.02            | -0.15        | -0.09             | +0.04         |

---

**Table 3: SUN, Food, ObjectNet**

| Method      | SUN B0 $\bar{A}$ | SUN B0 $A_B$ | SUN B150 $\bar{A}$ | SUN B150 $A_B$ | Food B0 $\bar{A}$ | Food B0 $A_B$ | Food B50 $\bar{A}$ | Food B50 $A_B$ | Obj B0 $\bar{A}$ | Obj B0 $A_B$ | Obj B100 $\bar{A}$ | Obj B100 $A_B$ |
| ----------- | ---------------- | ------------ | ------------------ | -------------- | ----------------- | ------------- | ------------------ | -------------- | ---------------- | ------------ | ------------------ | -------------- |
| Finetune    | +20.09           | +7.01        | +19.28             | +7.15          | +23.86            | +7.11         | +22.44             | +7.62          | +6.03            | +4.90        | +9.36              | +5.22          |
| SimpleCIL   | +0.02            | +0.02        | +0.03              | +0.02          | 0                 | +0.05         | +0.05              | +0.05          | 0                | 0            | 0                  | 0              |
| ZS-CLIP     | +0.03            | +0.03        | +0.03              | +0.03          | 0                 | +0.07         | +0.04              | +0.07          | 0                | 0            | 0                  | 0              |
| L2P         | -3.40            | **-6.80**    | **-17.38**         | **-27.49**     | -1.78             | -2.62         | +2.55              | +4.23          | +10.94           | +9.49        | +7.17              | +4.36          |
| DualPrompt  | +1.19            | +0.33        | -3.67              | **-7.88**      | +1.54             | +1.65         | +1.06              | +0.36          | +4.99            | +3.92        | +5.64              | +1.78          |
| CODA-Prompt | -1.87            | -3.46        | **-9.59**          | **-15.87**     | **-5.87**         | **-11.74**    | **-9.76**          | **-18.17**     | **+14.45**       | **+12.57**   | +9.31              | +1.64          |
| RAPF        | +2.89            | +5.14        | +3.31              | +4.97          | +1.65             | +3.75         | +2.18              | +3.56          | +5.15            | +8.22        | +6.22              | +6.44          |
| ENGINE      | -0.01            | -0.09        | +0.04              | -0.02          | -0.03             | +0.15         | +0.03              | 0              | +0.01            | -0.08        | -0.03              | -0.23          |

---

**Summary**

- **SimpleCIL / ZS-CLIP / ENGINE**: reproduced results are highly consistent with the papers, with differences within ±0.5.
- **Finetune**: reproduced results are **significantly higher** than the paper results, possibly due to different backbones or configurations.
- **CODA-Prompt**: reproduced results are **significantly lower** than the paper results on multiple B50/B100/B150 settings, with the largest gap being -20.90.
- **L2P / DualPrompt**: mixed behavior, with some settings higher and some lower, especially lower $A_B$ on B100/B50 settings.
- **RAPF**: mostly close to or slightly higher than the paper results, but clearly lower on CUB.

---

## CIFAR-100

> seed=1993/2 corresponds to different class-incremental orders.

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

## ImageNet-R

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

## CUB200

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
