from transformers.generation import LogitsProcessor  # HuggingFace的LogitsProcessor基类，用于修改生成时的logits
from transformers import AutoTokenizer  # 用于tokenizer相关操作
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union  # 类型提示
import math  # 数学运算
import numpy as np  # 数值计算
import torch  # PyTorch张量操作
import warnings  # 警告处理

from transformers.utils import add_start_docstrings  # 自动添加docstring装饰器

# =============================================================================
# Docstring: 定义LogitsProcessor的输入输出格式文档
# =============================================================================
LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            输入的token序列，shape为(批量大小, 序列长度)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search
            语言模型头部的预测分数，shape为(批量大小, 词汇表大小)
            每一行表示当前token位置对词汇表中每个词的logits预测

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.
            处理后的预测分数，shape不变，但某些token的分数被修改为负无穷
"""

# =============================================================================
# ConstrainedLogitsProcessor: 前缀约束解码处理器
# 功能: 在生成每个token时，根据已生成的前缀动态限制允许的下一个token
# 原理: 构建一个"前缀→允许token"的映射表，生成时只从允许的token中选择
# =============================================================================
class ConstrainedLogitsProcessor(LogitsProcessor):
    """
    前缀约束解码器，用于确保生成的token序列符合预定义的前缀规则。

    主要应用场景:
    - 确保生成的文本符合特定格式（如SID: <a_X><b_Y><c_Z>）
    - 防止生成无效的token组合
    - 通过约束搜索空间提高生成质量和效率

    工作流程:
    1. __init__: 初始化，设置约束函数和参数
    2. __call__: 每次生成token时被调用，过滤不允许的token
    """

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        # 回调函数：给定batch_id和当前已生成的token序列，返回允许的下一个token列表
        # 签名: (batch_id, input_ids) -> List[int]
        num_beams: int,
        # Beam Search的beam数量，用于正确处理多个beam的约束
        base_model: str = None,
        # 基础模型名称，用于判断模型类型（如qwen/llama/gpt2）
        eos_token_id: int = None
        # 序列结束token的id，用于检测何时无合法token可生成
    ):
        """
        初始化约束处理器。

        Args:
            prefix_allowed_tokens_fn: 前缀约束函数，输入(批量id, 当前token序列)，
                                       输出该位置允许的下一个token id列表
            num_beams: beam search的宽度
            base_model: 模型名称
            eos_token_id: 结束符id
        """
        # 保存约束函数引用
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        # 保存beam数量，用于后续处理多beam情况
        self._num_beams = num_beams
        # 计数器：记录当前是第几步生成（从0开始）
        self.count = 0
        # 保存基础模型名称
        self.base_model = base_model
        # 保存eos_token_id，用于无合法token时强制结束
        self.eos_token_id = eos_token_id

        # 根据模型类型确定prefix_index（"### Response:\n"对应的token数量）
        # GPT2使用4个token，其他模型（如Qwen）使用3个token
        if self.base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4  # GPT2的特殊前缀长度
        else:
            self.prefix_index = 3  # Qwen/Llama等模型的前缀长度

    # =========================================================================
    # __call__: 核心方法，每次生成token时被调用
    # 输入: input_ids（当前序列）, scores（原始logits）
    # 输出: 处理后的scores（将不允许的token设为负无穷）
    # =========================================================================
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        处理logits，应用前缀约束。

        核心逻辑:
        1. 对scores取log_softmax得到对数概率
        2. 创建mask，将所有token初始化为负无穷（禁止选择）
        3. 对每个beam，根据当前前缀查询允许的token
        4. 将允许的token在mask中设为0（允许选择）
        5. mask + scores = 最终分数（不允许的token概率被清零）

        Args:
            input_ids: 形状(batch_size, seq_len)，当前已生成的token序列
            scores: 形状(batch_size, vocab_size)，模型输出的原始logits

        Returns:
            处理后的scores，不允许的token对应位置为负无穷
        """
        # Step 1: 将logits转换为对数概率（log_softmax）
        # 这样后续可以直接通过加法应用mask
        scores = torch.nn.functional.log_softmax(scores, dim=-1)

        # Step 2: 创建mask，初始化为全负无穷（默认禁止所有token）
        # shape: (batch_size, vocab_size)
        mask = torch.full_like(scores, float('-inf'))

        # Step 3: 遍历所有batch和beam，应用约束
        # input_ids的shape是(batch_size * num_beams, seq_len)
        # 需要reshape回(batch_size, num_beams, seq_len)来分别处理每个beam
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                # Step 3a: 确定当前beam的"当前前缀"用于查询约束
                # 第一次生成(step 0): 使用固定前缀（"### Response:\n"之后的部分）
                # 后续生成: 使用已生成的token序列
                if self.count == 0:
                    # 初始状态：取固定长度(prefix_index)的token作为查询key
                    hash_key = sent[-self.prefix_index:]
                else:
                    # 后续状态：取最近生成的count个token作为查询key
                    hash_key = sent[-self.count:]

                # Step 3b: 将tensor转换为list，用于查询约束函数
                hash_key = hash_key.tolist()

                # Step 3c: 调用约束函数，获取该前缀允许的下一个token列表
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)

                # Step 3d: 检查是否有合法的下一个token
                if len(prefix_allowed_tokens) == 0:
                    # 无合法token，发出警告并强制生成EOS
                    warnings.warn(
                        f"No valid tokens found for hash_key {hash_key} at step {self.count}. "
                        f"This indicates the model generated an unexpected token. "
                    )
                    # 强制该beam只能生成EOS token
                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                    continue  # 跳过当前beam，继续处理下一个

                # Step 3e: 将允许的token在mask中设为0（0 + (-inf) = -inf，但后面的mask操作会覆盖）
                # 实际上：mask初始为-inf，我们想让允许的token不被mask，所以设为0
                # 但下面的 scores = scores + mask 会将-inf的位置保持为-inf
                # 所以正确的逻辑是：mask[allowed] = 0，其他保持-inf
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        # Step 4: 更新步数计数器
        self.count += 1

        # Step 5: 应用mask
        # scores + mask = 最终分数
        # 允许的token: -inf + 0 = 原分数（不变）
        # 禁止的token: -inf + (-inf) = -inf（概率为0）
        scores = scores + mask

        # Step 6: 返回处理后的scores
        return scores
