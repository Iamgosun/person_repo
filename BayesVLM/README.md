# Iamgosun/person_repo 项目 README

## Executive Summary

## AI回复要求：返回给我的修改代码注释需要使用中文；返回的修改代码需要是可替换的完整代码；每次的修改都要严谨认真的检查修改是否正确合理
## 注意本项目是在BayesVLM项目代码上重新编写的，batch["text"]是源码的方式，但我的CoOp与适配器不需要使用它，而是另外取。并且适配器还没有写完
# BayesVLM（当前实验版）README

## 1. 项目概述

本仓库是在 **BayesVLM** 思路基础上做的实验性改造版本，核心目标不是从零训练视觉 backbone，而是：

- 使用 **冻结的 CLIP / SigLIP 图像编码器与文本编码器**
- 在 few-shot 场景下，只对**很小的一部分参数**做训练
- 对比两条主实验路线：
  1. `text_only_bayes_coop`：文本侧贝叶斯 CoOp
  2. `vlm_adapter`：冻结 VLM backbone 上的轻量 adapter 训练
- 统一保存训练日志、metrics、checkpoint、逐样本预测结果，方便后续分析

当前仓库已经形成两条可以直接运行的主训练链：

- `train_py/train_text_only_bayes_coop.py`
- `train_py/train_vlm_adapter.py`

对应的 shell 启动脚本为：

- `train_py/run_text_only_bayes_coop.sh`
- `train_py/run_vlm_adapter.sh`

---

## 2. 这两个任务分别是什么

### 2.1 任务 A：`text_only_bayes_coop`

这是当前仓库里**更偏 BayesVLM 主线**的方法。

它的目标是：

- 冻结 `image_encoder`
- 冻结 `text_encoder`
- 冻结 `vlm`（相似度头）
- 只训练 **CoOp 的上下文 prompt 参数**
- 用预先估计好的 **文本投影层 Hessian**，构造文本侧后验协方差
- 最终输出带不确定性的 `ProbabilisticLogits(mean, var)`

这条方法的核心思想是：

> 图像特征保持确定性；
> 文本侧类别原型由 CoOp prompt 生成；
> 文本投影层的不确定性通过 Hessian + Kronecker 因子传播到分类 logits 上。

它适合回答的问题是：

- 在 few-shot 条件下，只调 prompt，能不能提升分类性能？
- 文本侧贝叶斯化之后，预测的 `NLPD` / `ECE` 是否更稳定？
- 在冻结 backbone 的前提下，prompt learning 能否带来比 zero-shot 更好的结果？

---

### 2.2 任务 B：`vlm_adapter`

这是当前仓库里的**对比实验主线**。

它的目标是：

- 冻结 `image_encoder`
- 冻结 `text_encoder`
- 冻结 `vlm`
- 只训练一层轻量 adapter
- 在固定的文本 prototype 基础上，让 adapter 学会做 few-shot 分类修正

当前 adapter 路线支持的实验类型包括：

- `LP`
- `TR`
- `CLIPA`
- `TIPA`
- `CROSSMODAL`
- `GAUSSIAN_PER_CLASS`

它适合回答的问题是：

- 和 prompt learning 比，adapter 在 few-shot 下是否更稳定？
- 不同 adapter 初始化方式对结果的影响如何？
- zero-shot baseline 到 adapter finetune 的增益有多大？

---

## 3. 项目整体训练流程

无论是 `text_only_bayes_coop` 还是 `vlm_adapter`，训练大框架都类似：

1. 读取数据集
2. 构建 `raw_train_ds / val_ds / test_ds`
3. 从 `raw_train_ds` 中抽取 few-shot 子集作为 `train_ds`
4. 加载 CLIP / SigLIP backbone
5. 构建共享图像特征缓存（可选但强烈建议）
6. 构建方法专属模型
7. 训练
8. 验证
9. 保存：
   - `train.log`
   - `config.json`
   - `metrics_history.json`
   - `metrics_history.csv`
   - `best checkpoint`
   - `train/val/test predictions`
   - `summary.json`

---

## 4. few-shot 与 cache 的设计说明

这是本项目最容易误解的两个点。

### 4.1 few-shot 是怎么做的

few-shot 不是通过缩小整个数据集目录实现的，而是：

- 先保留完整训练集为 `raw_train_ds`
- 再从 `raw_train_ds` 中按 `shots_per_class` 抽样，得到真正训练用的 `train_ds`

也就是说：

- `train_ds` 是 few-shot 子集
- `val_ds` 和 `test_ds` 仍然是完整验证/测试集

这也是为什么日志里会看到：

- `train` 每类只有 1 张或 16 张
- `test` 每类依然很多张

这是**正常设计**，不是 few-shot 失效。

---

### 4.2 image feature cache 是怎么做的

本项目的缓存不是缓存整个训练状态，而是缓存：

- 图像 embedding
- 图像激活
- 图像 residual
- 类别 id
- sample key
- manifest

缓存目录会根据这些条件生成：

- 数据集名
- split（`train_full / val / test`）
- 模型名
- 本地权重路径
- 图像尺寸
- transform 名字
- data_root

缓存的意义是：

- 第一次运行：提取图像特征并落盘
- 第二次运行：直接加载 `.pt/.json` 文件，不再重复走 image encoder

特别注意：

- 当前 few-shot 训练通常缓存的是 **`train_full`**，不是 few-shot 后的小训练集
- few-shot 子集是从 `train_full` 特征里按索引再裁出来的

所以你在 1-shot 场景下，第一次缓存阶段仍然可能看到较大的 `train_full` 进度条，这是正常的。

---

## 5. 目录结构总览

下面是当前项目建议理解方式下的目录结构：

```text
BayesVLM/
├── README.md
├── bayesvlm/
│   ├── common.py
│   ├── constants.py
│   ├── utils.py
│   ├── precompute.py
│   ├── hessians.py
│   ├── coop_prompt.py
│   ├── text_only_bayes_coop.py
│   ├── vlm_adapter.py
│   ├── adapter.py
│   ├── text_priors.py
│   ├── image_encoder.py
│   ├── text_encoder.py
│   ├── vlm.py
│   ├── semantic_postnet.py
│   ├── flows.py
│   ├── data/
│   │   ├── common.py
│   │   ├── dataset_ops.py
│   │   ├── factory.py
│   │   ├── pipeline.py
│   │   ├── cifar10.py
│   │   ├── cifar100.py
│   │   ├── food101.py
│   │   ├── flowers102.py
│   │   ├── sun397.py
│   │   ├── ucf101.py
│   │   ├── homeoffice.py
│   │   ├── homeoffice_da.py
│   │   ├── imagenet_1k.py
│   │   ├── imagenet_wds.py
│   │   ├── imagenet_r.py
│   │   ├── imagenet_sketch.py
│   │   ├── imagenet_da.py
│   │   └── laion400m.py
│   ├── features/
│   │   ├── image_cache.py
│   │   └── feature_dataset.py
│   ├── methods/
│   │   ├── text_only_bayes_coop.py
│   │   └── vlm_adapter.py
│   └── training/
│       ├── io.py
│       ├── runtime.py
│       ├── history.py
│       └── metrics.py
├── train_py/
│   ├── train_text_only_bayes_coop.py
│   ├── train_vlm_adapter.py
│   ├── run_text_only_bayes_coop.sh
│   └── run_vlm_adapter.sh
├── scripts/
│   ├── hessian_estimation.py
│   └── hf/
├── datasets/
├── models/
├── hessians/
├── cache/
├── output/
└── output_adapter/