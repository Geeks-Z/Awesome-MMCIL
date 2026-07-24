# 📊 评估指标与实验结果说明

[中文](实验结果.md) | [English](实验结果.en.md)

## 🛠️ 评估指标

> - [Continual learning for VLMs: a survey and taxonomy beyond forgetting](http://arxiv.org/abs/2508.04227)
> - [Recent advances of multimodal continual learning: a comprehensive survey](http://arxiv.org/abs/2410.05352)


<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20251203104009.png" style="zoom: 60%;" /></div>

> 记 $p^j_i$ 为“模型在完成第 $i$ 个任务训练后（$i=0$ 表示预训练状态），在第 $j$ 个任务上的性能/准确率”。



**Average Accuracy** 是基础评估指标，用于衡量所有学习阶段的平均任务性能，从而反映模型的整体能力。

> 对于传统 continual learning，只有对角线以上的分数有意义，因为传统模型无法对未见任务进行 zero-shot 预测。$\bar{\mathcal{A}} = \frac{1}{T}\sum_{i=1}^{T}( \frac{1}{i} \sum_{j=1}^{i}p_i^j)$

$$
Avg=\frac{1}{T} \sum_{t=1}^{T} \left(\frac{1}{T} \sum_{i=1}^{T} p_t^{(i)}\right)
$$

**Last Accuracy** 衡量模型在完成全部训练后的保留能力，反映实际部署时的可用性能。
$$
\textbf{Last} = \frac{1}{T} \sum_{t=1}^{T}p^t_T
$$

**Forgetting Ratio** 衡量每个任务在初次学习后出现的最大性能下降。
$$
\textbf{Forget}=\frac{1}{T-1} \sum_{i=1}^{T-1} \max_{i \leq j \leq T-1}\left( p_j^{i} - p_T^{i} \right)
$$

**Backward Transfer (BWT)** 衡量后续任务学习对先前任务造成的提升或退化。

> <div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20260127095257.png" style="zoom: 80%;" /></div>
>
> 就是使用模型在学习最后一个（第 $T$ 个）task以后对之前第 $i$ 个task的表现减去刚刚学完第i个时候的表现的差值(通常为负数，上图中的蓝色部分)。反应的是记忆能力！
>
> - 如果这个差值很大，就意味着模型对于之前学会的知识忘记的很多；
> - 如果这个差值很小，就意味着模型对于之前学会的知识忘记的很少；
> - 如果这个差值大于零，就意味着以后学到的知识对于之前学过的是一个促进的作用。

$$
\textbf{BWT}=\frac{1}{T-1} \sum_{t=1}^{T-1} \left(p_t^{(T)} -p_t^{(t)}  \right)
$$

**Forward Transfer (FWT)** 

> <div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20260127095811.png" style="zoom: 80%;" /></div>
>
> 我们以最后一个任务task $T$ 为例，这个参数反应的是机器还没有学习 $T$，只是学习了 $T$ 之前别的task时候对于 $T$ 的影响。我们用机器学习之前 $T-1$ 个task时候对任务task $T$ 的表现减去初始化参数时候对task $T$ 的表现(下图绿色部分)。用这个参数衡量机器学习之前任务对现在任务的影响。如果这个参数是正的，那么就说明这个机器是会触类旁通的。反应的是**迁移能力**！
> $$
> FWT = \frac{1}{T-1}\sum_{i=2}^TR_{i-1,i}-R_{0,i}
> $$

**Zero-shot Transfer** 衡量模型利用已有知识泛化到未见任务的能力。
$$
\textbf{Transfer} =\frac{1}{T-1} \sum_{t=2}^{T} \left(\frac{1}{t-1} \sum_{i=1}^{t-1} p_t^{(i)}\right)
$$



**Zero-Shot Degradation** 显式衡量 zero-shot 能力的退化，这是 VLM 在 continual learning 中的重要风险。
$$
\textbf{ZSD}=\frac{1}{T-1} \sum_{t=2}^{T} \max_{1 \leq i \leq t-1}\left( p_t^{(1)} - p_t^{(i)} \right)
$$



**Recall@K 和 mean Average Precision** 
$$
R@K=\frac{|\mathcal{R}_q\cap\{ d_1, d_2,\ldots,d_K\} |}{|\mathcal{R}_q|}\\
    mAP= \frac{1}{Q} \sum_{i=1}^{Q} \frac{1}{m_q} \sum_{k=1}^{K} P_q(k) \delta_q(k)
$$

---

## 📝 复现实验说明

### PROOF和ENGINE说明

1. PROOF：这个方法本身利用了memory，所以在复现L2P，DualPrompt，CODA-Prompt等方法时，也对这些方法添加了memory

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20260301102903.png" style="zoom: 80%;" /></div>

2. ENGINE：所有方法没有memory

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20260301102957.png" style="zoom: 80%;" /></div>

---

### C3Box代码修改

1. 配置文件注释掉memory（L2P，DualPrompt，CODA-Prompt）

2. `models/l2p.py`注释掉对memory的调用

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

3. `models/l2p.py:` `_init_train`方法，取消 `logits[:, :self._known_classes] = float('-inf')`的注释，减少对旧类分类器权重的更改

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

## 📊 复现结果

### 实验设置

- ### 数据集划分

  - B-$m$ Inc-$n$ ：$m$代表初始增量阶段类别数量，$n$ 代表后续每个增量阶段的类别数量
  
- ### 骨干网络

  | 模型                   | 训练数据                               | 发布方             | 特点                                 | 模型规模                       | 创建方式                                                     |
  | ---------------------- | -------------------------------------- | ------------------ | ------------------------------------ | ------------------------------ | ------------------------------------------------------------ |
  | **OpenAI CLIP**        | OpenAI 内部收集的 4 亿图文对数据集     | OpenAI             | 官方版本，性能稳定，但训练数据不公开 | ViT-B/32, ViT-B/16, ViT-L/14等 | `model, preprocess = clip.load("ViT-B/32", device=device)`   |
  | **OpenCLIP LAION400M** | 公开的 LAION-400M 数据集（4 亿图文对） | LAION 组织开源项目 | 完全开源，训练数据公开可获取，可复现 | 多种架构和规模选择             | `model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')` |

  > 均采用`OpenAI_CLIP`

- ### Memory

  对于 exemplar 参数，DER、iCaRL 和 FOSTER 在 CIFAR100 上将 `fixed_memory` 设为 false，并保持 `memory_size` 为 2000；在 ImageNet-R 上将 `fixed_memory` 设为 true，并保持 `memory_per_class` 为 20。其他方法不使用 exemplars。

### 实验结果

当前日志结果以 [MMCL_Baselines.xlsx](MMCL_Baselines.xlsx) 为准。
