# 九种方法的参数量与效率报告

三次完整独立运行的正式统计及与 SAGLA、SAGLA-C 的合并表见
[`COMBINED_REPEATED_REPORT.md`](COMBINED_REPEATED_REPORT.md)。论文应优先引用该表；
下方 Cars/Aircraft 表保留首轮逐次记录，作为原始审计材料。

## 建议采用的论文报告规则

不要把不同含义的量压缩成一个含糊的 `Params`。主表至少拆成：

1. `Peak updated params`：所有任务和训练阶段中，任一 optimizer step 前实际具有梯度的唯一参数元素数的峰值；
2. `Final extra learned params`：相对一份共同冻结 OpenCLIP，在最小最终推理导出中仍需额外保留的学习参数；
3. `Extra frozen params`：共同 backbone 之外的冻结编码器或实现副本；
4. `Auxiliary state`：原型、均值、协方差、特征统计和历史张量，按实际 dtype 以 MiB 报告；
5. 关于类别数 `C`、任务数 `T` 的增长式。

共同 backbone 为 OpenCLIP ViT-B/16 LAION-400M（149.62M 参数），不重复计入 `Final extra learned params`。`D=512` 为 CLIP 嵌入维度，`V=768` 为视觉投影前维度，CLG-CBM 的每类概念数 `p=10`。经重参数化的权重只有在替换原权重且验证导出等价后，才能按净额外参数计数。

## 参数量（C=100，T=10）

| 方法 | 峰值实际更新 (M) | 最终额外学习参数 (M) | 辅助状态 (MiB) | 额外冻结参数 (M) | 随 `C,T` 增长 |
|---|---:|---:|---:|---:|---|
| ZS-CLIP | 0.00 | 0.00 | 0.00 | 0.00 | `O(1)`；可选文本缓存 `O(CD)` |
| L2P | 0.09 | 0.09 | 0.00 | 86.19 | 固定 prompt pool + `O(CD)` head |
| DualPrompt | 0.29 | 0.29 | 0.00 | 86.19 | 固定 prompt pool + `O(CD)` head |
| CODA-Prompt | 3.64 | 3.64 | 0.00 | 0.00 | 固定分配 prompt pool + `O(CD)` head |
| SimpleCIL | 0.00 | 0.00 | 0.20 | 0.00 | `O(CD)` prototypes |
| RAPF | 0.26 | 0.26 | 100.46 | 0.00 | `O(D^2)` adapter + `O(CD^2)` statistics |
| ENGINE | 0.53 | 5.25 | 3.03 | 149.62* | `O(TD^2)` adapters + `O(CV+V^2)` statistics |
| CLG-CBM | 0.61 | 0.61 | 111.00 | 0.00 | `O(pCD+pC^2)` learned + `O(CD^2)` statistics |
| BOFA | 0.40 | 0.00 | 70.46 | 0.00 | 融合后净额外 `O(1)`；状态 `O(TV^2+TVD+CV)` |

关键解释：

- ZS-CLIP 无优化参数；如缓存文本特征，应另报 `O(CD)` 缓存。
- SimpleCIL 的 `100 x 512` 类原型由均值估计得到，不算梯度学习参数，FP32 状态为 0.20 MiB。
- L2P 和 DualPrompt 还需要一份 86.19M 参数的冻结视觉查询编码器。
- ENGINE 当前构造函数意外保留一份未注册、未使用的完整 OpenCLIP（149.62M）；它是可裁剪的实现冗余，不属于算法统计状态。
- BOFA 训练时实际更新投影与当前任务头；只有融合投影替换原 CLIP 投影并裁掉最终 `_eval_cnn` 未使用的任务头后，才是 0 个净额外部署参数。

## 数据集与计时协议

九个本地数据集中，按训练图像数最小的两个是 Cars（4,135 train / 4,083 test）和 Aircraft（6,667 train / 3,333 test），均按 B0-Inc10、100 类、10 任务运行。

硬件为同一节点的 NVIDIA A800 80GB PCIe，任务分配到物理 GPU 0/1/2；backbone/checkpoint 和 seed（1993）统一。训练时间是十个任务 `incremental_train` 的 CUDA 同步墙钟总和；包含算法必需的统计、伪样本和模型选择，排除模型加载、任务级评测以及仅用于日志的 epoch 测试。CLG-CBM 的逐 epoch 验证参与选模，因此保留。训练使用各方法原配置的 batch 和 epoch 日程。

推理是在最终 100 类完整测试集上统一 batch 64、8 workers；常规最终评测用于预热，随后执行 3 次完整 `_eval_cnn`，报告中位数、范围和 images/s。它包含数据加载、文本/统计准备及方法特有重排。显存为进程内 `torch.cuda.max_memory_allocated`，不采用整卡 `nvidia-smi`。

本次统计实验的有效运行日志按方法归档在 `logs/OpenCLIP_LAION400M_ViTB16/<方法>/efficiency/`；启动器日志位于 `logs/OpenCLIP_LAION400M_ViTB16/queue/efficiency/`，无效尝试审计日志位于对应方法的 `efficiency/audit/`，SAGLA 系列日志位于 `logs/OpenCLIP_LAION400M_ViTB16/SAGLA/efficiency/`。

## Cars

| 方法 | 训练设置 | 训练时间 (min) | 训练峰值 (GiB) | 推理中位数 [min-max] (s) | images/s | 推理峰值 (GiB) |
|---|---|---:|---:|---:|---:|---:|
| ZS-CLIP | 无 optimizer | —（0.28 s setup） | 0.57 | 11.53 [11.37-11.66] | 354.15 | 1.04 |
| L2P | bs16, 10 ep/task | 7.28 | 2.85 | 21.95 [21.94-22.04] | 186.02 | 1.39 |
| DualPrompt | bs64, 10 ep/task | 6.46 | 6.95 | 20.69 [20.53-20.69] | 197.38 | 1.25 |
| CODA-Prompt | bs128, 10 ep/task | 7.80 | 15.53 | 21.92 [21.77-22.12] | 186.31 | 0.97 |
| SimpleCIL | bs48, prototype estimation | 0.37 | 0.92 | 10.93 [10.81-11.18] | 373.43 | 1.04 |
| RAPF | bs128, 10 ep/task | 4.62 | 7.17 | 13.89 [13.75-13.97] | 293.96 | 4.57 |
| ENGINE | bs64, 10 ep/task | 7.33 | 1.11 | 144.76 [144.39-145.20] | 28.21 | 2.89 |
| CLG-CBM | bs48, 60 ep/task + selection | 27.50 | 1.05 | 11.09 [10.98-11.15] | 368.14 | 1.16 |
| BOFA | bs128, 15+2 ep/task | 7.44 | 1.73 | 20.28 [20.23-20.31] | 201.31 | 1.13 |

## Aircraft

| 方法 | 训练设置 | 训练时间 (min) | 训练峰值 (GiB) | 推理中位数 [min-max] (s) | images/s | 推理峰值 (GiB) |
|---|---|---:|---:|---:|---:|---:|
| ZS-CLIP | 无 optimizer | —（0.20 s setup） | 0.57 | 10.74 [9.86-11.00] | 310.37 | 1.04 |
| L2P | bs16, 10 ep/task | 11.61 | 2.85 | 18.83 [18.79-18.83] | 177.01 | 1.39 |
| DualPrompt | bs64, 10 ep/task | 10.45 | 6.95 | 17.69 [17.65-17.69] | 188.46 | 1.25 |
| CODA-Prompt | bs128, 10 ep/task | 12.58 | 15.53 | 18.39 [18.36-18.41] | 181.20 | 0.97 |
| SimpleCIL | bs48, prototype estimation | 0.73 | 0.92 | 11.54 [9.88-13.97] | 288.90 | 1.04 |
| RAPF | bs128, 10 ep/task | 7.77 | 7.17 | 11.61 [11.59-11.80] | 287.10 | 4.57 |
| ENGINE | bs64, 10 ep/task | 8.98 | 1.07 | 119.44 [119.00-119.54] | 27.91 | 2.89 |
| CLG-CBM | bs48, 60 ep/task + selection | 32.16 | 1.05 | 9.82 [9.63-11.24] | 339.27 | 1.16 |
| BOFA | bs128, 15+2 ep/task | 11.96 | 1.88 | 17.04 [16.90-17.13] | 195.61 | 1.12 |

三次完整系统运行现已完成。正式报告采用三次独立运行的均值、样本方差和
标准差；每次运行内部的三次推理仅用于取得该次运行的中位数，不能当作三个
独立实验样本。

## 结果解读与推荐表述

- 参数高效不等于显存高效：CODA-Prompt 只更新 3.635M 参数，但训练峰值为 15.53 GiB；RAPF 只更新 0.262M 参数，训练峰值仍为 7.17 GiB。
- 状态量不能塞进参数量：RAPF、CLG-CBM、BOFA 分别保留约 100.455、110.997、70.457 MiB 的实现状态。
- ENGINE 的最终端到端推理仅约 28 images/s，明显慢于其他方法；原因是其 `_eval_cnn` 包含逐 batch 的多轮描述编码和重排，因此不应只报告 backbone latency。
- SimpleCIL 的训练成本最低；CLG-CBM 因每任务 60 epoch 和验证选模，完整训练时间最长。比较训练时间时必须同时给出日程。

论文中可采用如下表注：

> `Peak updated params` denotes the maximum number of unique parameters with non-null gradients at any optimizer step. `Final extra learned params` counts method-specific learned parameters retained by a minimal inference export, excluding one common frozen OpenCLIP ViT-B/16. Frozen encoder copies and non-parametric continual-learning state are reported separately. Training time is one synchronized end-to-end run over all ten tasks; inference is the median of three full final-stage test passes after warm-up. GPU memory is process-local peak allocated memory.
