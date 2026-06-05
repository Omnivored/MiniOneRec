<div align="center">

<img src="./assets/logo.png" width="500em" ></img>

**一个用于扩展生成式推荐的开源框架**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg) <a href="https://arxiv.org/abs/2510.24431"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red"></a>

<a href="https://arxiv.org/abs/2510.24431">📄 技术报告</a> | <a href="https://huggingface.co/kkknight/MiniOneRec">🤗 Huggingface</a> | <a href="https://modelscope.cn/models/k925238839/MiniOneRec">🤖  Modelscope</a>

</div>

**MiniOneRec** 是首个完全开源的**生成式推荐**框架，它提供了一个端到端的工作流程，涵盖 **SID 构建**、**监督微调 (SFT)** 和面向推荐的**强化学习 (RL)**。

***

## 📢 公告

- 2026-01-04 — 关于基于 Instruct 模型的复现结果与我们报告的指标之间可能存在的差异，请检查评估日志中的 CC 指标是否非零（参见 calc.py）。如果非零，表明模型仍在生成大量无效项，约束解码尚未成功。我们怀疑此问题可能与 transformer 等依赖项的版本有关，我们仍在调查原因以提供通用解决方案。同时，您可以将 Instruct 模型切换到 base 模型，例如 Qwen2.5-base，以避免此问题。
- 2025-12-04 — 我们更新了新脚本来支持处理 Amazon23 数据集。
- 2025-12-01 — 我们修复了 data.py 中的一个 bug，该 bug 可能导致 SID-item 对齐任务提前看到答案。这是因为我们之前尝试使用部分轨迹来引导完整的 SID-item 生成，不影响模型性能。
- 2025-11-20 — **RQ-Kmeans+** 中的 SID 构建方法已更新（首次在 **GPR** 中提出，这是首次开源复现）。
- 2025-11-19 — 我们基于 Accelerate 实现了多 GPU 并行 text-to-embedding 方法，效率显著高于原始版本：rq/text2emb/amazon\_text2emb.py
- 2025-11-19 — **constrained-RQ-Kmeans** 中的 SID 构建方法已更新。
- 2025-11-07 — 感谢提交 issue！根据您的反馈，我们发布了新的实现。如果在运行代码时遇到任何问题，请先更新并查阅**最新版本**。
- 2025-11-07 — 您现在可以选择在 SFT 阶段冻结 LLM 参数，仅训练新添加的 SID 词汇的 embeddings。
- 2025-10-31 — 您现在可以直接下载我们 MiniOneRec 模型的实施 **checkpoints**。
- 2025-10-31 — **RQ-Kmeans** 中的 SID 构建方法已更新。

***

## 🛠️ 关键技术

<div align="center">
<img src="./assets/minionerec_framework.png" width=100% ></img> 
</div>

- **SID 构建：MiniOneRec 首先将每个产品转换为紧凑的、具有语义意义的 token。** 它连接项目的 title 和 description，将这个句子输入冻结的 text encoder，然后用三级 RQ-VAE 量化生成的 embedding。
- **SFT：在将所有项目重写为 SID 后，模型首先以监督方式进行训练。** 它将按时间顺序排列的用户历史视为 token 序列，并通过 next-token prediction 学习生成用户可能消费的下一个项目的 SID。至关重要的是，此阶段与一组语言对齐目标共同训练，这些目标在自然语言和 SID 空间之间进行映射，使推荐系统能够继承嵌入大型语言模型中的世界知识，同时将这些知识扎根于离散的项目代码中。
- **面向推荐的 RL：在 SFT 之后，MiniOneRec 通过基于 GRPO 的面向推荐的 RL 阶段进一步优化。** 为每个 prompt 生成多个候选推荐，在组内标准化它们的 rewards 以稳定 gradients，KL penalty 使更新后的 policy 保持接近其 reference。由于 action space 是项目 SIDs 的封闭列表，系统切换到 constrained beam search，保证每个 beam 都是唯一且有效的，大大提高了 sampling 效率和 diversity。reward signal 本身融合了 binary correctness term 和 rank-aware component，后者对高概率但不正确的项目施加更重的惩罚，并可以用 collaborative-filtering scores 增强。该 pipeline 使 MiniOneRec 能够结合密集的语言知识，实现高性能、轻量级的生成式推荐系统。

***

## 📊 评估

<div align="center">
<img src="./assets/minionerec_main_result.png" width=100% ></img> 
</div>

***

## 🗂️ 仓库概览

| 文件 / 目录                              | 描述                                                                                  |
| ------------------------------------ | ----------------------------------------------------------------------------------- |
| `sft.sh`                             | 启动 Supervised Fine-Tuning (SFT) 阶段的 Shell 脚本                                        |
| `sft.py`                             | SFT training loop 的 Python 实现                                                       |
| `sft_gpr.py`                         | GPR 启发的 SFT，具有 Value-Aware Fine-Tuning (VAFT)：基于模拟 item value 实现 weighted loss      |
| `rl.sh`                              | 启动 Reinforcement Learning (RL) 阶段的 Shell 脚本                                         |
| `rl.py`                              | RL training loop 的 Python 实现                                                        |
| `rl_gpr.py`                          | GPR 启发的 RL，具有 Hierarchy Enhanced Policy Optimization (HEPO)                         |
| `minionerec_trainer.py`              | MiniOneRec trainer — 基于 GRPO 的 trainer，专用于生成式推荐                                     |
| `configs/`                           | YAML 配置文件                                                                           |
| `evaluate.sh`                        | 一键式离线 Top-K 评估脚本                                                                    |
| `evaluate.py`                        | 用于计算 HR\@K 和 NDCG\@K 的评估工具。                                                         |
| `LogitProcessor.py`                  | 用于 constrained decoding 的 Logit processor（Python 实现）                                |
| `data.py`                            | 用于 SFT 和 RL 训练的 Data pipeline                                                       |
| `convert_dataset.py`                 | 将 RQ 训练的数据集转换为 SFT-then-RL 格式                                                       |
| `convert_dataset_gpr.py`             | GPR 启发的数据集转换器：注入模拟的 heterogeneous tokens (U/E/I/O) 以模拟 unified input representation |
| `data/amazon18_data_process.sh`      | 将 Amazon18 数据过滤和预处理为 RQ 就绪格式的 Shell 脚本                                              |
| `data/amazon18_data_process.py`      | Amazon18 数据预处理 pipeline 的 Python 实现                                                 |
| `data/amazon18_data_process_gpr.py`  | GPR 启发的 Amazon18 预处理：提取 heterogeneous features 以用于 unified input representation     |
| `data/amazon23_data_process.sh`      | 将 Amazon23 数据过滤和预处理为 RQ 就绪格式的 Shell 脚本                                              |
| `data/amazon23_data_process.py`      | Amazon23 数据预处理 pipeline 的 Python 实现                                                 |
| `rq/text2emb/amazon_text2emb.sh`     | 通过 emb\_model 为 Amazon 数据集生成 item embeddings (title + description) 的 Shell 脚本       |
| `rq/text2emb/amazon_text2emb.py`     | 上述 embedding 生成的 Python 实现                                                          |
| `rq/text2emb/amazon_text2emb_gpr.py` | GPR 启发的 text-to-embedding                                                           |
| `rq/generate_indices.py`             | 在训练 RQ-VAE 模型后生成 SID 文件                                                             |
| `rq/rqvae.sh`                        | 在 Amazon item embeddings 上训练 RQ-VAE 的 Shell 脚本                                      |
| `rq/rqvae.py`                        | RQ-VAE 训练的 Python 实现                                                                |
| `rq/rqkmeans_faiss.py`               | 基于 faiss 的 RQ-Kmeans 训练的 Python 实现                                                  |
| `rq/rqkmeans_constrained.py`         | Constrained RQ-Kmeans 的 Python 实现                                                   |
| `rq/rqkmeans_constrained.sh`         | 在 Amazon item embeddings 上训练 constrained RQ-Kmeans 的 Shell 脚本                       |
| `rq/rqkmeans_plus.py`                | RQ-Kmeans+ 的 Python 实现                                                              |
| `rq/rqkmeans_plus.sh`                | 在 Amazon item embeddings 上训练 RQ-Kmeans+ 的 Shell 脚本                                  |
| `rq/generate_indices_plus.py`        | 在训练 RQ-Kmeans+ 模型后生成 SID 文件                                                         |
| `rq/generate_indices_plus.sh`        | 在训练 RQ-Kmeans+ 模型后生成 SID 文件的 Shell 脚本                                               |
| `requirements.txt`                   | Python 依赖项列表                                                                        |

***

## 🚀 快速开始

使用我们提供的预训练 Industrial/Office SIDs 快速开始！
只需 4-8 个 A100/H100 GPUs 即可复现。

### 1. 创建独立的 Python 环境

```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```

### 2. 安装所需的包

```bash
pip install -r requirements.txt
```

### 3. SFT

```bash
bash sft.sh
```

### 4. 面向推荐的 RL

```bash
bash rl.sh
```

### 5. 运行评估 bash

```bash
bash evaluate.sh
```

***

## 📜 完整 Pipeline 详解

### 0. 前置条件

- GPUs: <例如，4-8 × A100/H100 80 GB 或同等配置>
- Python: 3.11

### 1. 环境设置

- **1.1 克隆仓库**

```
git clone https://github.com/AkaliKong/MiniOneRec.git
cd MiniOneRec
```

- **1.2 创建并激活 conda 环境**

```
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```

- **1.3 安装依赖项**

```
pip install -r requirements.txt
```

### 2. 数据准备

- **2.1 下载原始数据集（可选）**\
  从官方页面获取：
  [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/),
  [Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/),
  [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)。
  注意：Industrial 和 Office 数据集包含在 Amazon 2018 中；Amazon 2014 和 2023 版本需要对我们的 data/amazon18\_data\_process.py 进行轻微修改。
- **2.2 过滤和预处理**

```
bash data/amazon18_data_process.sh \
     --dataset  /mnt/c/Users/omni/Desktop/下载/Industrial_and_Scientific.csv # 例如 Industrial
     --user_k 5 \
     --item_k 5 \
     --st_year 2017 \
     --st_month 10 \
     --ed_year 2018 \
     --ed_month 11 \
     --output_path ./data/Amazon18
```

- **2.3 将项目文本编码为 embeddings**

```
bash rq/amazon_text2emb.sh \
     --dataset your_dataset_type \ # 例如，Industrial 
     --root your_processed_dataset_path \
     --plm_name qwen \
     --plm_checkpoint your_emb_model_path
```

### 3. SID 构建

选择 3.1.1、3.1.2、3.1.3 或 3.1.4。

- **3.1.1 在 embeddings 上训练 RQ-VAE**

```
bash rq/rqvae.sh \
      --data_path xxx/data/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
      --ckpt_dir ./output/Industrial_and_Scientific \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480
```

- **3.1.2 在 embeddings 上训练 RQ-Kmeans**

```
conda install faiss-gpu
python rqkmeans_faiss.py --dataset Industrial_and_Scientific # 基于语义 embeddings 的 RQ-Kmeans 方法具有相对较高的冲突率。
```

- **3.1.3 在 embeddings 上训练 constrained RQ-Kmeans**
  对于冲突的项目，我们添加额外的层来执行去重；同时，我们使用平衡约束来确保 SIDs 均匀分布。

```
pip install k_means_constrained
pip install polars
bash rqkmeans_constrained.sh
```

- **3.1.4 在 embeddings 上训练 RQ-Kmeans+**

```
pip install k_means_constrained
pip install polars
bash rqkmeans_constrained.sh
bash rqkmeans_plus.sh
```

- **3.2 生成 indices（仅需 RQ-VAE 和 RQ-Kmeans+）**

```
python rq/generate_indices.py
# 或
bash rq/generate_indices_plus.sh
```

- **3.3 转换数据集格式**

```
python convert_dataset.py \
     --dataset_name Industrial_and_Scientific \
     --data_dir /mnt/workspace/minionerec/data/Amazon18/Industrial_and_Scientific \
     --output_dir /mnt/workspace/minionerec/data/Amazon18

```

### 4. SFT

```
bash sft.sh \
     --base_model your_model_path \
     --output_dir your_ourput_dir \
     --sid_index_path your_.index.json_path \
     --item_meta_path your_.item.json_path
```

### 5. 面向推荐的 RL

> （可选）对于生产规模的数据集，考虑到强化学习的成本和边际收益递减，您可以仅使用数万样本量的相对较小的子集来执行 RL 阶段。

```
bash rl.sh \
     --model_path your_model_path \
     --output_dir output_dir \
```

### 6. 离线评估

```
bash evaluate.sh \
     --exp_name your_model_path 
```

***

## 🤖 支持的 LLM 提供商

MiniOneRec 支持多种 LLM 提供商用于文本丰富任务（例如，用户偏好和项目特征提取）。在您的 `api_info` 字典中配置提供商：

| 提供商                                 | `provider` 值 | 默认 Base URL                 | 示例模型                           |
| ----------------------------------- | ------------ | --------------------------- | ------------------------------ |
| OpenAI                              | `"openai"`   | —                           | `text-davinci-003`             |
| DeepSeek                            | `"deepseek"` | `https://api.deepseek.com`  | `deepseek-chat`                |
| [MiniMax](https://www.minimaxi.com) | `"minimax"`  | `https://api.minimax.io/v1` | `MiniMax-M2.7`, `MiniMax-M2.5` |

**示例 — 使用 MiniMax：**

```python
api_info = {
    "provider": "minimax",
    "api_key_list": ["your-minimax-api-key"],
    "base_url": "https://api.minimax.io/v1",  # optional, this is the default
}
get_res_batch("MiniMax-M2.7", prompt_list, max_tokens=512, api_info=api_info)
```

***

## 📝 即将推出的功能

我们正在积极扩展 MiniOneRec 的功能。以下增强功能已在我们的路线图上：

- ⏱️ **更多 SID 构建算法**：即将支持 R-VQ、RQ-Kmeans、RQ-OPQ 和 RQ-VAE-v2 (PLUM)。
- ⚙️ **MiniOneRec-Think**：一个无缝集成对话、推理和个性化推荐的模块，为复杂的交互式场景提供一体化解决方案。
- 🔍 **更广泛的数据集支持**：额外的流行公共数据集，包括 Yelp，以进一步验证我们算法的通用性。

***

## 🏫 研究机构  <!-- omit in toc -->

本项目由以下机构开发：

- <img src="assets/lds.png" width="28px"> [LDS](https://data-science.ustc.edu.cn/_upload/tpl/15/04/5380/template5380/index.html)
- <img src="assets/alphalab.jpg" width="28px"> [AlphaLab](https://alphalab-ustc.github.io/index.html)
- <img src="assets/next.jpg" width="28px"> [NExT](https://www.nextcenter.org/)

***

## 🧩 贡献

我们欢迎并感谢所有贡献！如果您有改进 MiniOneRec 的想法，请随时提交 pull request (PR)。

***

## 🙏 致谢

本仓库复用或改编了以下开源项目的部分代码。我们衷心感谢这些项目的作者和贡献者：

- [ReRe](https://github.com/sober-clever/ReRe)
- [LC-Rec](https://github.com/zhengbw0324/LC-Rec)

***

## 🔖 引用 <!-- omit in toc -->

如果您发现我们的代码/论文/模型有帮助，请考虑引用我们的论文 📝 并给我们加星 ⭐️！

```bib
@misc{MiniOneRec,
      title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation}, 
      author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
      year={2025},
      eprint={2510.24431},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
}

@article{ReRe,
      title={Reinforced Preference Optimization for Recommendation}, 
      author={Junfei Tan and Yuxin Chen and An Zhang and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Xiang Wang},
      journal={arXiv preprint arXiv:2510.12211},
      year={2025},
}

@inproceedings{RecZero,
      title={Think before Recommendation: Autonomous Reasoning-enhanced Recommender}, 
      author={Xiaoyu Kong and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Jiancan Wu and Xiang Wang},
      year={2025},
      booktitle={NeurIPS},
}

```

***

<div align="center">
我们欢迎来自社区的贡献！🤝
</div>
