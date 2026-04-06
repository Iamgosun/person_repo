# Iamgosun/person_repo 项目 README

## Executive Summary

## AI回复要求：返回给我的修改代码注释需要使用中文；返回的修改代码需要是可替换的完整代码；每次的修改都要严谨认真的检查修改是否正确合理
## 注意本项目是在BayesVLM项目代码上重新编写的，batch["text"]是源码的方式，但我的CoOp与适配器不需要使用它，而是另外取。并且适配器还没有写完
本次研究**优先且首先使用已启用的连接器：GitHub（api_tool: github）**，并且**仅审查你指定的仓库 `Iamgosun/person_repo`** 的源码与脚本；
依照train_text_only_bayes_coop与train_vlm_adapter。这条方法本身的代码链思考作为主线
结论上，这个仓库的是围绕 **BayesVLM（ICLR 2026）** 思路做的实验性改造：用 CLIP/SigLIP 的图像/文本编码器（多为冻结），通过 **Hessian→投影层后验协方差** 做不确定性传播，再在下游进行零样本评估或轻量训练（如 CoOp soft prompt、或实验性的 PostNet 式密度证据头）。
# BayesVLM（当前重构版）项目说明

> 这是一个围绕 **Bayesian Vision-Language Model** 做自己的实验train_text_only_bayes_coop，以及
**Vision-Language Model**做另外的对比实验train_vlm_adapter。
> 当前代码以两条主训练线为核心：
>
> 1. `text_only_bayes_coop`：文本侧贝叶斯 CoOp
> 2. `vlm_adapter`：基于冻结 VLM backbone 的轻量 adapter 训练
>
> 项目的目标不是从零训练一个 CNN，而是：
>
> - 使用冻结的 CLIP / SigLIP 图像与文本编码器
> - 在文本侧或类别原型侧做轻量训练
> - 评估 few-shot 场景下的分类性能
> - 保存 checkpoint、metrics、逐样本预测结果，便于后续分析

---

## 1. 项目当前任务概览

当前项目主要做两件事：

### 自身任务 A：`text_only_bayes_coop`
目标：
- 冻结 image encoder / text encoder / VLM 头
- 只训练 CoOp prompt context（soft prompt）
- 结合 Hessian 估计得到文本投影层后验协方差
- 在 few-shot 场景下做贝叶斯化的文本侧分类

特点：
- 依赖 `hessian_dir`
- 输出是 `ProbabilisticLogits`
- 训练对象很小，主要是 prompt 参数
- 当前在 `cifar10` 上已经闭环跑通

---

### 对比任务 B：`vlm_adapter`
目标：
- 冻结 image encoder / text encoder / VLM backbone
- 在类别文本 prototype 基础上训练 adapter
- 支持多种 adapter：
  - LP
  - TR
  - CLIPA
  - TIPA
  - CROSSMODAL
  - GAUSSIAN_PER_CLASS

特点：
- 不依赖 Hessian 主链路
- 输出是普通 logits
- 支持 zero-shot baseline + adapter finetune
- 当前工程链路已经跑通，但具体实验效果依赖初始化方式和训练配置

---

## 2. 项目整体流程

无论哪条训练线，整体流程都是：

1. 读取数据集
2. 做 few-shot 抽样
3. 构建 dataloader
4. 加载 CLIP / SigLIP backbone
5. 构建方法专属模型
6. 训练
7. 验证 / 测试
8. 保存：
   - config
   - metrics_history
   - best checkpoint
   - train / val / test predictions
   - summary

---

## 3. 代码目录结构

```text
BayesVLM/
├── bayesvlm/                  # 核心 Python 包
│   ├── data/                  # 数据集、DataModule、数据准备流程
│   ├── training/              # 训练公共工具（日志、metrics、runtime）
│   ├── methods/               # 两条主训练线的方法私有逻辑
│   ├── adapter.py             # 各类 adapter 定义
│   ├── common.py              # 通用数据结构（如 EncoderResult、ProbabilisticLogits）
│   ├── constants.py           # 模型名称映射、常量配置
│   ├── coop_prompt.py         # CoOp prompt learner
│   ├── flows.py               # flow / density 建模模块（供 PostNet 等实验用）
│   ├── hessians.py            # Hessian / Kronecker 协方差 / λ 优化
│   ├── image_encoder.py       # 图像编码器封装
│   ├── semantic_postnet.py    # PostNet 风格头（实验模块）
│   ├── text_encoder.py        # 文本编码器封装
│   ├── text_only_bayes_coop.py# 文本侧贝叶斯 CoOp 模型定义
│   ├── text_priors.py         # 文本模板 / 文本先验构造
│   ├── utils.py               # 模型加载、transform 选择等公共函数
│   ├── vlm.py                 # CLIP / SIGLIP VLM 头
│   └── vlm_adapter.py         # VLMAdapter 主模型封装
│
├── train_py/                  # 训练入口脚本与 shell 启动脚本
│   ├── train_text_only_bayes_coop.py
│   ├── train_vlm_adapter.py
│   ├── run_text_only_bayes_coop.sh
│   └── run_vlm_adapter.sh
│
├── scripts/                   # 额外脚本，如 Hessian 估计
│   └── hessian_estimation.py
│
├── datasets/                  # 本地数据目录（不一定纳入版本管理）
├── hessians/                  # Hessian 结果目录（不一定纳入版本管理）
├── models/                    # 本地缓存的 CLIP/SigLIP 权重
├── output/                    # text_only_bayes_coop 输出
├── output_adapter/            # vlm_adapter 输出
└── README.md