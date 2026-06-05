
import torch
import random
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset


def generate_completion(model, tokenizer, prompt, max_new_tokens=256):
    """Generate completion for a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def get_samples_from_sid_dataset(num_samples, train_file, tokenizer, item_meta_path, sid_index_path, category):
    """Get samples from SidSFTDataset"""
    dataset = SidSFTDataset(
        train_file=train_file,
        tokenizer=tokenizer,
        max_len=2048,
        sample=-1,
        seed=42,
        category=category
    )
    
    # Get random indices
    indices = random.sample(range(len(dataset.data)), min(num_samples, len(dataset.data)))
    
    samples = []
    for idx in indices:
        # Get raw data
        row = dataset.data.iloc[idx]
        history = dataset.get_history(row)
        
        # Build the full prompt
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        
        history_for_prompt = history.copy()
        history_for_prompt['output'] = ''
        prompt = instruction + dataset.generate_prompt(history_for_prompt)
        
        label = history['output']
        
        samples.append({
            'dataset': 'SidSFTDataset',
            'prompt': prompt,
            'label': label,
            'history': history
        })
    
    return samples


def get_samples_from_itemfeat_dataset(num_samples, tokenizer, item_meta_path, sid_index_path, category):
    """Get samples from SidItemFeatDataset"""
    dataset = SidItemFeatDataset(
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=2048,
        sample=-1,
        category=category
    )
    
    # Get random indices
    indices = random.sample(range(len(dataset.data)), min(num_samples, len(dataset.data)))
    
    samples = []
    for idx in indices:
        # Get raw data
        data_point = dataset.data[idx]
        
        # Build the full prompt
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Answer the question about item identification.

"""
        
        prompt = instruction + dataset.generate_prompt(data_point)
        label = data_point['output'] + '\n'
        
        samples.append({
            'dataset': 'SidItemFeatDataset',
            'task': data_point['task'],
            'prompt': prompt,
            'label': label,
            'data_point': data_point
        })
    
    return samples


def get_samples_from_fusion_dataset(num_samples, train_file, tokenizer, item_meta_path, sid_index_path, category):
    """Get samples from FusionSeqRecDataset"""
    dataset = FusionSeqRecDataset(
        train_file=train_file,
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=2048,
        sample=-1,
        category=category
    )
    
    # Get random indices
    indices = random.sample(range(len(dataset.data)), min(num_samples, len(dataset.data)))
    
    samples = []
    for idx in indices:
        # Get raw data
        history_data = dataset.get_history(dataset.data.iloc[idx])
        
        # Build the full prompt
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you recommend the next item for the user based on their interaction history?

"""
        
        prompt_text = dataset.generate_prompt_title(history_data['history_str'])
        formatted_prompt = dataset.generate_formatted_prompt(prompt_text, "")
        
        prompt = instruction + formatted_prompt
        label = history_data['target_title'] + '\n'
        
        samples.append({
            'dataset': 'FusionSeqRecDataset',
            'prompt': prompt,
            'label': label,
            'history_data': history_data
        })
    
    return samples


def main():
    # Configuration
    model_path = "hy-tmp/MiniOneRec/output/checkpoint-312"
    train_file = "data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"
    item_meta_path = "data/Amazon/Industrial_and_Scientific/Industrial_and_Scientific.item.json"
    sid_index_path = "data/Amazon/Industrial_and_Scientific/Industrial_and_Scientific.index.json"
    category = "Industrial and Scientific"
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # Get samples from each dataset
    num_samples = 10
    print(f"\nGetting {num_samples} samples from each dataset...")
    
    # Get samples from SidSFTDataset
    sid_samples = get_samples_from_sid_dataset(
        num_samples,
        train_file,
        tokenizer,
        item_meta_path,
        sid_index_path,
        category
    )
    
    # Get samples from SidItemFeatDataset
    itemfeat_samples = get_samples_from_itemfeat_dataset(
        num_samples,
        tokenizer,
        item_meta_path,
        sid_index_path,
        category
    )
    
    # Get samples from FusionSeqRecDataset
    fusion_samples = get_samples_from_fusion_dataset(
        num_samples,
        train_file,
        tokenizer,
        item_meta_path,
        sid_index_path,
        category
    )
    
    # Combine all samples
    all_samples = sid_samples + itemfeat_samples + fusion_samples
    
    # Generate and evaluate
    print("\nGenerating completions and comparing with labels...")
    print("=" * 100)
    
    for i, sample in enumerate(all_samples):
        print(f"\nSample {i+1}/{len(all_samples)}")
        print(f"Dataset: {sample['dataset']}")
        
        if 'task' in sample:
            print(f"Task: {sample['task']}")
        
        print("\nPrompt:")
        print("-" * 50)
        print(sample['prompt'])
        print("-" * 50)
        
        # Generate completion
        generated = generate_completion(model, tokenizer, sample['prompt'])
        
        # Print results
        print(f"\nLabel: {repr(sample['label'])}")
        print(f"Generated: {repr(generated)}")
        
        # Check if match (ignoring trailing newlines and whitespace)
        label_clean = sample['label'].strip()
        generated_clean = generated.strip()
        
        if label_clean == generated_clean:
            print("✅ MATCH!")
        else:
            print("❌ NO MATCH")
        
        print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
