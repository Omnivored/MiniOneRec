#!/bin/bash

export NCCL_IB_DISABLE=1
export WANDB_MODE=online
export WANDB_API_KEY='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'
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    model_path=./output_dir/checkpoint-3300
    output_dir=./output_dir
    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --config_file ./config/single_gpu.yaml \
                                    --num_processes 1 --main_process_port 29503 \
                                    rl.py \
                        --model_path ${model_path} \
                        --resume_from_checkpoint ${model_path} \
                        --train_batch_size 8 \
                        --eval_batch_size 16 \
                        --num_train_epochs 2 \
                        --gradient_accumulation_steps 2 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train False \
                        --eval_step 0.1 \
                        --reward_type ranking \
                        --num_generations 4 \
                        --mask_all_zero False \
                        --dynamic_sampling False \
                        --sync_ref_model True \
                        --beam_search True \
                        --test_during_training False \
                        --temperature 1.0 \
                        --learning_rate 1e-5 \
                        --add_gt False \
                        --beta 1e-3 \
                        --dapo False \
                        --output_dir ${output_dir} \
                        --wandb_run_name "" \
                        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
                        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
done
