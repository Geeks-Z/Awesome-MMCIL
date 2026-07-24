# 📊 Evaluation Metrics and Reproduction Results

[中文](实验结果.md) | [English](实验结果.en.md)

## 🛠️ Evaluation Metrics

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

**Forward Transfer (FWT)**

> <div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20260127095811.png" style="zoom: 80%;" /></div>
>
> Taking the final task $T$ as an example, this metric reflects how learning the previous $T-1$ tasks affects performance on task $T$ before task $T$ itself is learned. It subtracts the model's initial performance on task $T$ from its performance on task $T$ after learning the first $T-1$ tasks, shown as the green region in the figure. A positive value indicates positive transfer from previous tasks and reflects **transfer ability**.
>
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

## 📝 Reproduction Notes

### Notes on PROOF and ENGINE

1. PROOF uses memory by design, so memory is also added when reproducing methods such as L2P, DualPrompt, and CODA-Prompt.

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20260301102903.png" style="zoom: 80%;" /></div>

2. ENGINE uses no memory for any method.

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20260301102957.png" style="zoom: 80%;" /></div>

---

### 🛠️ C3Box Code Changes

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

### PromptFusion

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

## 📊 Reproduced Results

### Experimental Setup

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

### Experiment Results

The current log-derived results are maintained in [MMCL_Baselines.xlsx](MMCL_Baselines.xlsx).
