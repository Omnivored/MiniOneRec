# SFT.py 详解

## 📋 概述

`sft.py` 是 MiniOneRec 框架的**监督微调（Supervised Fine-Tuning）**阶段，用于将预训练的 LLM（如 Qwen2.5-0.5B）适配到推荐任务。

---

## 🔧 核心功能

### 1. 数据加载与处理
- 加载用户行为序列数据（训练集、验证集）
- 加载商品元数据和 SID 索引
- Tokenizer 扩展（添加 SID tokens）

### 2. 模型初始化
- 加载预训练 LLM 模型
- 可选：冻结 LLM 参数，只训练新增 embedding

### 3. 训练配置
- 设置学习率、优化器、梯度累积
- 配置 Early Stopping
- 配置 bf16 混合精度

### 4. 训练与评估
- 多数据集融合训练
- 每 5% 步数进行评估
- 保存最佳模型

---

## 📊 训练数据集

| 数据集 | 类 | 说明 |
|--------|-----|------|
| `SidSFTDataset` | 用户序列 | 用户历史行为 → 预测下一个 SID |
| `SidItemFeatDataset` | 商品特征 | 学习 SID ↔ 自然语言 映射 |
| `FusionSeqRecDataset` | 融合训练 | 结合用户偏好 + 商品特征 |

---

## 🚀 运行命令

```bash
torchrun --nproc_per_node 1 sft.py \
    --base_model ./Qwen2.5-0.5B \
    --batch_size 128 \
    --micro_batch_size 4 \
    --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --output_dir ./output/Qwen2.5-0.5B/Industrial_and_Scientific \
    --wandb_project MiniOneRec \
    --wandb_run_name Industrial_sft \
    --category Industrial_and_Scientific \
    --train_from_scratch False \
    --seed 42 \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --freeze_LLM False
```

---

## 📝 终端输出详解

### 阶段 1：初始化

```
Industrial_and_Scientific
```

**含义**：打印类别名称（转换为自然语言描述）

---

### 阶段 2：模型加载

```
Loading index from ./data/Amazon/index/Industrial_and_Scientific.index.json
Adding 3686 new tokens to tokenizer
```

**含义**：
| 输出 | 说明 |
|------|------|
| `Loading index` | 加载 SID 索引文件 |
| `Adding 3686 new tokens` | 向 tokenizer 添加 3686 个 SID token |

---

### 阶段 3：数据集加载

```
LOAD DATA FINISHED
Dataset({
    features: ['input_ids', 'labels', 'attention_mask'],
    num_rows: 36259
})
Dataset({
    features: ['input_ids', 'labels', 'attention_mask'],
    num_rows: 4532
})
```

**含义**：
| 输出 | 说明 |
|------|------|
| `LOAD DATA FINISHED` | 数据集加载完成 |
| 训练集 | 36259 条样本 |
| 验证集 | 4532 条样本 |

---

### 阶段 4：训练开始

```
{'loss': 0.2156, 'grad_norm': 1.234, 'learning_rate': 1.5e-05, 'epoch': 0.05}
{'loss': 0.1892, 'grad_norm': 0.987, 'learning_rate': 3.0e-05, 'epoch': 0.10}
{'loss': 0.1765, 'grad_norm': 0.876, 'learning_rate': 4.5e-05, 'epoch': 0.15}
...
```

**含义**：
| 字段 | 说明 |
|------|------|
| `loss` | 当前 step 的训练损失 |
| `grad_norm` | 梯度范数（监控梯度爆炸） |
| `learning_rate` | 当前学习率（逐渐上升） |
| `epoch` | 等效 epoch 数（0.05 = 5%） |

---

### 阶段 5：评估

```
{'eval_loss': 1.818, 'eval_runtime': 23.2, 'eval_samples_per_second': 194.9, 'eval_steps_per_second': 12.2, 'epoch': 5.5}
```

**含义**：
| 字段 | 说明 |
|------|------|
| `eval_loss` | 验证集损失（约 1.8） |
| `eval_runtime` | 评估耗时（秒） |
| `eval_samples_per_second` | 评估速度 |
| `epoch` | 当前 epoch |

---

### 阶段 6：训练完成

```
100%|██████████| 858/858 [02:03:50<00:00, 8.26s/it]
train_runtime: 7435.158
train_samples_per_second: 107.374
train_steps_per_second: 0.21
train_loss: 0.5604
epoch: 5.5
```

**含义**：
| 字段 | 说明 |
|------|------|
| `train_runtime` | 总训练时间（秒）≈ 2 小时 |
| `train_samples_per_second` | 训练吞吐量 |
| `train_loss` | 平均训练损失 |
| `epoch` | 完成 5.5 个 epoch |

---

## 📂 保存的文件

```
./output/Qwen2.5-0.5B/Industrial_and_Scientific/
├── checkpoint-858/              # 最后一个 checkpoint
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
├── final_checkpoint/           # 最佳模型（loss 最低）
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
└── trainer_state.json          # 训练状态
```

---

## ⚙️ 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 128 | 总批次大小 |
| `micro_batch_size` | 4 | 每个 GPU 的批次大小 |
| `num_epochs` | 10 | 训练轮数 |
| `learning_rate` | 3e-4 | 学习率 |
| `cutoff_len` | 512 | 最大序列长度 |
| `freeze_LLM` | False | 是否冻结 LLM |
| `eval_step` | 0.05 | 评估频率（每 5% 步数） |
| `save_total_limit` | 1 | 最多保存 1 个 checkpoint |

---

## 🔄 训练流程图

```
┌─────────────────────────────────────────────────────────┐
│  1. 环境设置                                             │
│     - 设置随机种子                                        │
│     - 设置 wandb 项目                                    │
│     - 计算 gradient_accumulation_steps                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  2. 模型加载                                             │
│     - 加载预训练 LLM (Qwen2.5-0.5B)                      │
│     - 加载/扩展 tokenizer (添加 SID tokens)               │
│     - 可选：冻结 LLM 参数                               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  3. 数据加载                                             │
│     - SidSFTDataset (用户序列)                           │
│     - SidItemFeatDataset (商品特征)                      │
│     - FusionSeqRecDataset (融合数据)                     │
│     - ConcatDataset (合并数据集)                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  4. 训练循环                                             │
│     for epoch in range(num_epochs):                     │
│         for step in training_steps:                      │
│             - 前向传播                                   │
│             - 计算 loss                                 │
│             - 反向传播                                   │
│             - 梯度累积 (gradient_accumulation_steps)     │
│             - 参数更新                                  │
│             - 日志输出 (每 step)                        │
│             - 评估 (每 5% 步数)                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  5. 保存模型                                             │
│     - 保存 final_checkpoint (最佳模型)                    │
│     - 保存 tokenizer                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📈 训练时间估算

| 配置 | batch_size | micro_batch_size | 预计时间 |
|------|-----------|-----------------|---------|
| 默认 | 128 | 4 | ~4-5 小时 |
| 优化01 | 256 | 8 | ~3-4 小时 |
| 优化02 | 512 | 16 | ~3 小时 |

**取决于 GPU 性能和数据集大小**

---

## ⚠️ Early Stopping

```python
callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
```

- 如果连续 3 次评估 loss 没有改善，训练自动停止
- 保存的是最佳模型（eval_loss 最低的）

---

## 🔍 关键机制

### 1. Token Extender
```python
token_extender = TokenExtender(data_path, dataset)
new_tokens = token_extender.get_new_tokens()  # 获取所有 SID tokens
tokenizer.add_tokens(new_tokens)  # 添加到 tokenizer
model.resize_token_embeddings(len(tokenizer))  # 扩展 embedding 层
```

### 2. 多数据集融合
```python
train_datasets = []
train_datasets.append(SidSFTDataset(...))        # 用户序列
train_datasets.append(SidItemFeatDataset(...))  # 商品特征
train_datasets.append(FusionSeqRecDataset(...)) # 融合
train_data = ConcatDataset(train_datasets)      # 合并
```

### 3. 梯度累积
```python
gradient_accumulation_steps = batch_size // micro_batch_size
# batch_size=128, micro_batch_size=4 → gradient_accumulation_steps=32
```
