
"""
SFT模型HitRate评估脚本
对train、valid、test三个数据集分别计算HR@K指标
"""

import torch
import json
import pandas as pd
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from data import SidSFTDataset
from LogitProcessor import ConstrainedLogitsProcessor
from torch.utils.data import DataLoader
import random


def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)


def evaluate_hitrate(
    model,
    tokenizer,
    dataset,
    info_file,
    num_beams=50,
    batch_size=8,
    max_new_tokens=64,
    device="cuda"
):
    """
    评估模型在给定数据集上的HitRate
    
    Args:
        model: 微调后的模型
        tokenizer: tokenizer
        dataset: SidSFTDataset数据集
        info_file: info文件路径（包含所有商品的SID）
        num_beams: Beam Search的束宽
        batch_size: 批处理大小
        device: 设备
    
    Returns:
        dict: 包含HR@K和NDCG@K指标的字典
    """
    topk = [1, 3, 5, 10, 20, 50]
    hr = [0] * len(topk)
    ndcg = [0] * len(topk)
    
    # 加载info文件构建约束
    with open(info_file, 'r') as f:
        info = f.readlines()
        semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
    
    info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]
    
    if "llama" in str(type(model)).lower() or "llama" in str(tokenizer.name_or_path).lower():
        prefix_index = 3
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
    else:
        prefix_index = 3
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
    
    # 构建hash_dict
    hash_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])
    
    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []
    
    # 获取所有样本的target
    targets = []
    prompts = []
    for i in range(len(dataset.data)):
        row = dataset.data.iloc[i]
        history = dataset.get_history(row)
        target_item = history['output'].strip() + "\n"
        targets.append(target_item)
        
        # 构建prompt
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        history_for_prompt = history.copy()
        history_for_prompt['output'] = ''
        prompt = instruction + dataset.generate_prompt(history_for_prompt)
        prompts.append(prompt)
    
    # Tokenize
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # 转换为列表格式进行批处理
    num_samples = len(prompts)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    results = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        
        batch_encodings = {
            "input_ids": encodings["input_ids"][start_idx:end_idx].to(device),
            "attention_mask": encodings["attention_mask"][start_idx:end_idx].to(device)
        }
        
        # 获取该batch的targets
        batch_targets = targets[start_idx:end_idx]
        
        # 生成配置
        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=0.0,
            num_return_sequences=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        
        # 约束处理器
        clp = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            base_model=str(tokenizer.name_or_path),
            eos_token_id=tokenizer.eos_token_id
        )
        
        with torch.no_grad():
            generation_output = model.generate(
                batch_encodings["input_ids"],
                attention_mask=batch_encodings["attention_mask"],
                generation_config=generation_config,
                logits_processor=[clp],
            )
        
        # 解码
        prompt_length = batch_encodings["input_ids"].shape[1]
        completions = tokenizer.batch_decode(
            generation_output[:, prompt_length:],
            skip_special_tokens=True
        )
        
        # 按beam分组
        completions_per_sample = [
            completions[i * num_beams:(i + 1) * num_beams]
            for i in range(len(completions) // num_beams)
        ]
        
        # 计算HR和NDCG
        for i, comp_lis in enumerate(completions_per_sample):
            target = batch_targets[i]
            for j, comp in enumerate(comp_lis):
                if comp.strip("\n\"") == target.strip("\n\""):
                    for idx, k in enumerate(topk):
                        if j < k:
                            hr[idx] += 1
                            ndcg[idx] += 1 / math.log2(j + 2)
                    break
    
    # 归一化
    hr = [h / num_samples for h in hr]
    ndcg = [n / num_samples for n in ndcg]
    
    return {
        "num_samples": num_samples,
        "HR@1": hr[0],
        "HR@3": hr[1],
        "HR@5": hr[2],
        "HR@10": hr[3],
        "HR@20": hr[4],
        "HR@50": hr[5],
        "NDCG@1": ndcg[0],
        "NDCG@3": ndcg[1],
        "NDCG@5": ndcg[2],
        "NDCG@10": ndcg[3],
        "NDCG@20": ndcg[4],
        "NDCG@50": ndcg[5],
    }


def main():
    # 配置
    model_path = "hy-tmp/MiniOneRec/output/checkpoint-312"  # SFT微调后的模型路径
    base_model_path = "./Qwen2.5-0.5B"  # 原始模型路径（用于tokenizer）
    
    # 数据集路径
    train_file = "data/origin00/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"
    valid_file = "data/origin00/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv"
    test_file = "data/origin00/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"
    
    info_file = "data/origin00/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"
    item_meta_path = "data/Amazon/Industrial_and_Scientific/Industrial_and_Scientific.item.json"
    sid_index_path = "data/Amazon/Industrial_and_Scientific/Industrial_and_Scientific.index.json"
    
    category = "Industrial and Scientific"
    
    # 评估参数
    num_beams = 50
    batch_size = 4
    max_new_tokens = 64
    
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载模型和tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集
    print("\nCreating datasets...")
    train_dataset = SidSFTDataset(
        train_file=train_file,
        tokenizer=tokenizer,
        max_len=512,
        sample=-1,
        seed=42,
        category=category
    )
    
    valid_dataset = SidSFTDataset(
        train_file=valid_file,
        tokenizer=tokenizer,
        max_len=512,
        sample=-1,
        seed=42,
        category=category
    )
    
    test_dataset = SidSFTDataset(
        train_file=test_file,
        tokenizer=tokenizer,
        max_len=512,
        sample=-1,
        seed=42,
        category=category
    )
    
    print(f"Train dataset size: {len(train_dataset.data)}")
    print(f"Valid dataset size: {len(valid_dataset.data)}")
    print(f"Test dataset size: {len(test_dataset.data)}")
    
    # 评估
    results = {}
    
    # Test集
    print("\n" + "=" * 60)
    print("Evaluating on TEST dataset...")
    print("=" * 60)
    results["test"] = evaluate_hitrate(
        model, tokenizer, test_dataset, info_file,
        num_beams=num_beams, batch_size=batch_size,
        max_new_tokens=max_new_tokens, device=device
    )
    
    # Valid集
    print("\n" + "=" * 60)
    print("Evaluating on VALID dataset...")
    print("=" * 60)
    results["valid"] = evaluate_hitrate(
        model, tokenizer, valid_dataset, info_file,
        num_beams=num_beams, batch_size=batch_size,
        max_new_tokens=max_new_tokens, device=device
    )
    
    # Train集（可以采样评估）
    print("\n" + "=" * 60)
    print("Evaluating on TRAIN dataset (sampled 10000 samples)...")
    print("=" * 60)
    # 为了加快评估，对训练集采样
    train_sampled_dataset = SidSFTDataset(
        train_file=train_file,
        tokenizer=tokenizer,
        max_len=512,
        sample=10000,  # 采样10000条
        seed=42,
        category=category
    )
    results["train"] = evaluate_hitrate(
        model, tokenizer, train_sampled_dataset, info_file,
        num_beams=num_beams, batch_size=batch_size,
        max_new_tokens=max_new_tokens, device=device
    )
    
    # 打印结果
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\n{'Dataset':<10} {'Samples':<10} {'HR@1':<8} {'HR@3':<8} {'HR@5':<8} {'HR@10':<8} {'NDCG@10':<10}")
    print("-" * 70)
    
    for dataset_name in ["train", "valid", "test"]:
        r = results[dataset_name]
        print(f"{dataset_name:<10} {r['num_samples']:<10} "
              f"{r['HR@1']:.4f}    {r['HR@3']:.4f}    {r['HR@5']:.4f}    {r['HR@10']:.4f}    {r['NDCG@10']:.4f}")
    
    # 保存结果
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
