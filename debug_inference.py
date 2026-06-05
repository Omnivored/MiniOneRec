# """
# 逐步推理观察脚本：展示从第一次 model(input_ids) 到最终输出推荐SID的完整过程
# 用法: python debug_inference.py --base_model <模型路径> --info_file <info文件路径> --test_data_path <测试CSV路径>
# 示例: python debug_inference.py --base_model ./Qwen2.5-0.5B --info_file ./data/origin00/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt --test_data_path ./data/origin00/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv
# """

# import torch
# import fire
# import json
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from LogitProcessor import ConstrainedLogitsProcessor


# def get_hash(x):
#     x = [str(_) for _ in x]
#     return '-'.join(x)


# def analyze_sid_tokenization():
#     """【新增功能】分析三层SID会被tokenizer映射成几个token"""
#     print("=" * 80)
#     print("【附加功能】三层SID的Tokenization分析")
#     print("=" * 80)

#     tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-0.5B")

#     sample_sids = [
#         "<a_223><b_80><c_216>",
#         "<a_1><b_2><c_3>",
#         "<a_236><b_231><c_226>",
#         "<a_99><b_99><c_99>",
#     ]

#     print("\nSID Tokenization分析结果：")
#     print("-" * 80)
#     print(f"{'SID':<25} {'Token数量':<10} {'Token IDs':<40} {'Token列表'}")
#     print("-" * 80)

#     for sid in sample_sids:
#         tokens = tokenizer.encode(sid, return_tensors="pt")[0]
#         token_texts = tokenizer.convert_ids_to_tokens(tokens.tolist())

#         print(f"{sid:<25} {len(tokens):<10} {str(tokens.tolist()):<40} {token_texts}")

#     print("\n详细分析（以第一个SID为例）：")
#     print("-" * 80)
#     sid = sample_sids[0]
#     tokens = tokenizer.encode(sid, return_tensors="pt")[0]
#     token_texts = tokenizer.convert_ids_to_tokens(tokens.tolist())

#     for i, (tid, ttext) in enumerate(zip(tokens.tolist(), token_texts)):
#         print(f"  Token {i+1}: ID={tid:>4}, Text='{ttext}'")

#     print(f"\n结论：三层SID '<a_X><b_Y><c_Z>' 被映射成 {len(tokens)} 个token")
#     print("       每个层级(<a_X>, <b_Y>, <c_Z>)分别被tokenize成多个子token")
#     print("       例如：'<a_223>' → ['<', 'a', '_', '2', '2', '3', '>'] (7个token)")

#     print("\n" + "=" * 80)


# def main(
#     base_model: str = "./Qwen2.5-0.5B",
#     info_file: str = "./data/origin00/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt",
#     test_data_path: str = "./data/origin00/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv",
#     max_new_tokens: int = 64,
#     num_beams: int = 5,
# ):
#     analyze_sid_tokenization()

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     print("=" * 80)
#     print("【第1步】加载模型和Tokenizer")
#     print("=" * 80)
#     tokenizer = AutoTokenizer.from_pretrained(base_model)
#     model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
#     model.eval()
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     tokenizer.padding_side = "left"
#     print(f"模型: {base_model}")
#     print(f"设备: {device}")
#     print(f"vocab_size: {len(tokenizer)}")
#     print(f"bos_token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
#     print(f"eos_token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
#     print()

#     print("=" * 80)
#     print("【第2步】构建约束Hash字典（从info文件中提取所有合法SID的token前缀）")
#     print("=" * 80)
#     with open(info_file, 'r') as f:
#         info = f.readlines()
#         semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
#         item_titles = [line.split('\t')[1].strip() + "\n" for line in info if len(line.split('\t')) >= 2]
#         item_id_list = [line.split('\t')[2].strip() for line in info if len(line.split('\t')) >= 3]

#     info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]

#     if base_model.lower().find("llama") > -1:
#         prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
#     else:
#         prefixID = [tokenizer(_).input_ids for _ in info_semantic]

#     if base_model.lower().find("gpt2") > -1:
#         prefix_index = 4
#     else:
#         prefix_index = 3

#     hash_dict = dict()
#     sid_to_title = {}
#     sid_to_itemid = {}
#     for idx, line in enumerate(info):
#         parts = line.strip().split('\t')
#         if len(parts) >= 2:
#             sid_to_title[parts[0]] = parts[1]
#         if len(parts) >= 3:
#             sid_to_itemid[parts[0]] = parts[2]

#     for index, ID in enumerate(prefixID):
#         ID.append(tokenizer.eos_token_id)
#         for i in range(prefix_index, len(ID)):
#             if i == prefix_index:
#                 hash_number = get_hash(ID[:i])
#             else:
#                 hash_number = get_hash(ID[prefix_index:i])
#             if hash_number not in hash_dict:
#                 hash_dict[hash_number] = set()
#             hash_dict[hash_number].add(ID[i])

#     for key in hash_dict.keys():
#         hash_dict[key] = list(hash_dict[key])

#     print(f"info文件中共有 {len(semantic_ids)} 个合法SID")
#     print(f"hash_dict中共有 {len(hash_dict)} 个前缀条目")
#     print(f"prefix_index = {prefix_index} (即'### Response:\\n'对应的token数)")
#     print(f"\n示例: 前5个SID:")
#     for i in range(min(5, len(semantic_ids))):
#         print(f"  {semantic_ids[i].strip()} -> {item_titles[i].strip()}")

#     print("\n【附加】hash_dict结构分析：")
#     print("-" * 80)
#     sample_keys = list(hash_dict.keys())[:5]
#     for key in sample_keys:
#         allowed = hash_dict[key]
#         allowed_texts = tokenizer.convert_ids_to_tokens(allowed)
#         print(f"  Key: '{key}' -> 允许的token: {allowed_texts} (IDs: {allowed})")
#         if len(allowed) <= 3:
#             for tid in allowed:
#                 print(f"    -> token '{tokenizer.decode([tid])}' (id={tid})")

#     print("\n" + "=" * 80)
#     print("【第3步】从测试数据构建输入Prompt")
#     print("=" * 80)
#     test_df = pd.read_csv(test_data_path)
#     row = test_df.iloc[0]

#     history_sids = eval(row['history_item_sid'])
#     target_sid = str(row['item_sid'])
#     target_title = str(row['item_title'])

#     history_str = ", ".join(history_sids)

#     instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

# ### Instruction:
# Can you predict the next possible item that the user may expect?

# """

#     user_input = f"""### User Input: 
# Can you predict the next possible item the user may expect, given the following chronological interaction history: {history_str}

# ### Response:\n"""

#     full_prompt = instruction + user_input

#     print(f"用户历史SID序列 ({len(history_sids)}个):")
#     for i, sid in enumerate(history_sids):
#         print(f"  [{i+1}] {sid}")
#     print(f"\n目标SID (真实值): {target_sid}")
#     print(f"目标Title (真实值): {target_title}")

#     print("\n【关键说明】target_item是SID格式，不是商品标题！")
#     print(f"  - target_item = '{target_sid}\\n' (SID格式)")
#     print(f"  - golden_tokens = tokenizer.encode('{target_sid}\\n')")
#     test_golden = tokenizer.encode(target_sid + "\n", return_tensors="pt")[0]
#     print(f"  - 对应 {len(test_golden)} 个token: {test_golden.tolist()}")
#     print(f"  - Token文本: {tokenizer.convert_ids_to_tokens(test_golden.tolist())}")

#     print(f"\n完整Prompt:")
#     print("-" * 40)
#     print(full_prompt)
#     print("-" * 40)

#     print("\n" + "=" * 80)
#     print("【第4步】Tokenizer编码")
#     print("=" * 80)

#     input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
#     attention_mask = torch.ones_like(input_ids).to(device)

#     print(f"input_ids 形状: {input_ids.shape}")
#     print(f"input_ids 内容 (前20个): {input_ids[0, :20].tolist()}")
#     print(f"对应文本 (前20个token): {tokenizer.convert_ids_to_tokens(input_ids[0, :20].tolist())}")
#     print(f"总token数: {input_ids.shape[1]}")
#     print()

#     print("=" * 80)
#     print("【第5步】逐步自回归生成（逐token观察）")
#     print("=" * 80)
#     print(f"max_new_tokens = {max_new_tokens}")
#     print(f"num_beams = {num_beams}")
#     print()

#     generated_ids = input_ids.clone()
#     step = 0

#     clp = ConstrainedLogitsProcessor(
#         prefix_allowed_tokens_fn=lambda batch_id, input_ids: (
#             prefix_allowed_tokens_fn(batch_id, input_ids) if 'prefix_allowed_tokens_fn' in dir() else []
#         ),
#         num_beams=1,
#         base_model=base_model,
#         eos_token_id=tokenizer.eos_token_id
#     )

#     def prefix_allowed_tokens_fn(batch_id, input_ids):
#         hash_number = get_hash(input_ids)
#         if hash_number in hash_dict:
#             return hash_dict[hash_number]
#         return []

#     clp._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn

#     print("开始生成...")
#     print("-" * 80)

#     with torch.no_grad():
#         while step < max_new_tokens:
#             outputs = model(input_ids=generated_ids, attention_mask=torch.ones_like(generated_ids))
#             logits = outputs.logits[:, -1, :]

#             scores = torch.nn.functional.log_softmax(logits, dim=-1)
#             mask = torch.full_like(scores, float('-inf'))

#             if step == 0:
#                 hash_key = generated_ids[0, -prefix_index:].tolist()
#             else:
#                 hash_key = generated_ids[0, -(step + 1):].tolist() if step < prefix_index else generated_ids[0, -clp.count:].tolist()

#             hash_key_str = get_hash(generated_ids[0, -prefix_index:].tolist() if step == 0 else generated_ids[0, -(clp.count if clp.count > 0 else 1):].tolist())
#             allowed_tokens = prefix_allowed_tokens_fn(0, generated_ids[0, -prefix_index:].tolist() if step == 0 else generated_ids[0, -(clp.count if clp.count > 0 else 1):].tolist())

#             if len(allowed_tokens) > 0:
#                 mask[0, allowed_tokens] = 0
#             constrained_scores = scores + mask

#             next_token_id = torch.argmax(constrained_scores, dim=-1).unsqueeze(-1)
#             next_token_text = tokenizer.decode(next_token_id[0])

#             top5_scores, top5_ids = torch.topk(constrained_scores[0], min(5, len(allowed_tokens)) if len(allowed_tokens) > 0 else 5)
#             top5_tokens = [tokenizer.decode([tid.item()]) for tid in top5_ids]

#             step += 1
#             clp.count = step

#             print(f"Step {step:3d} | 生成token: '{next_token_text}' (id={next_token_id.item()}) | "
#                   f"允许token数: {len(allowed_tokens)} | Top5候选: {list(zip(top5_tokens, top5_scores.tolist()))}")

#             generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

#             if next_token_id.item() == tokenizer.eos_token_id:
#                 print(f"\n>>> 在Step {step}生成EOS token，生成终止！")
#                 break

#     print("\n" + "=" * 80)
#     print("【第6步】解码生成结果")
#     print("=" * 80)

#     prompt_length = input_ids.shape[1]
#     completion_ids = generated_ids[0, prompt_length:]
#     completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

#     predicted_sid = completion_text.split("Response:\n")[-1].strip() if "Response:\n" in completion_text else completion_text.strip()

#     print(f"生成的token序列 (共{len(completion_ids)}个):")
#     print(f"  IDs: {completion_ids.tolist()}")
#     print(f"  Tokens: {tokenizer.convert_ids_to_tokens(completion_ids.tolist())}")
#     print(f"\n解码后的完整文本: {repr(completion_text)}")
#     print(f"\n提取的预测SID: {predicted_sid}")

#     if predicted_sid in sid_to_title:
#         print(f"SID对应的商品Title: {sid_to_title[predicted_sid]}")
#     if predicted_sid in sid_to_itemid:
#         print(f"SID对应的商品ID: {sid_to_itemid[predicted_sid]}")

#     print(f"\n真实目标SID: {target_sid}")
#     print(f"真实目标Title: {target_title}")
#     print(f"预测是否正确: {'✓ 正确' if predicted_sid == target_sid else '✗ 错误'}")

#     print("\n" + "=" * 80)
#     print("【第7步】对比：使用model.generate()的标准推理（beam search）")
#     print("=" * 80)

#     clp2 = ConstrainedLogitsProcessor(
#         prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
#         num_beams=num_beams,
#         base_model=base_model,
#         eos_token_id=tokenizer.eos_token_id
#     )
#     from transformers import GenerationConfig, LogitsProcessorList

#     generation_config = GenerationConfig(
#         num_beams=num_beams,
#         length_penalty=0.0,
#         num_return_sequences=num_beams,
#         pad_token_id=tokenizer.eos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         max_new_tokens=max_new_tokens,
#     )

#     with torch.no_grad():
#         generation_output = model.generate(
#             input_ids,
#             attention_mask=attention_mask,
#             generation_config=generation_config,
#             return_dict_in_generate=True,
#             output_scores=True,
#             logits_processor=LogitsProcessorList([clp2]),
#         )

#     batched_completions = generation_output.sequences[:, prompt_length:]
#     outputs_text = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
#     outputs_text = [_.split("Response:\n")[-1].strip() for _ in outputs_text]

#     print(f"Beam Search结果 (num_beams={num_beams}):")
#     for i, out in enumerate(outputs_text):
#         title = sid_to_title.get(out, "未知")
#         is_correct = "✓" if out == target_sid else "✗"
#         print(f"  Beam {i+1}: SID={out} | Title={title} | {is_correct}")

#     print(f"\n真实目标: SID={target_sid} | Title={target_title}")
#     print()


# if __name__ == '__main__':
#     fire.Fire(main)
import torch

input_ids = torch.rand(6, 6)
print(input_ids)
print(input_ids.shape)

num_beams = 3

print(input_ids.view(-1, num_beams, input_ids.shape[-1]))
print(input_ids.view(-1, num_beams, input_ids.shape[-1]).shape)
print("#############################################")

for batch_id, beam_sent in enumerate(input_ids.view(-1, num_beams, input_ids.shape[-1])):
    print(batch_id, '-----', beam_sent)
    for beam_id, sent in enumerate(beam_sent):
            print(beam_id, '---', sent)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")