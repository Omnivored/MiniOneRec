python evaluate_rqkmeans.py \
  --data_path ../data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
  --pretrained_codebook_path ../data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.codebooks_constrained.npz \
  --checkpoint_path Apr-24-2026_17-15-24/epoch_9999_collision_0.0944_model.pth \
  --num_emb_list 256 256 256 \
  --e_dim 896 \
  --batch_size 10240
