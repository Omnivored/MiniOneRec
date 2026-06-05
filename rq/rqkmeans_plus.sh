python rqkmeans_plus.py \
  --data_path ../data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
  --pretrained_codebook_path ../data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.codebooks_constrained.npz \
  --num_emb_list 256 256 256 \
  --e_dim 896 \
  --lr 1e-5 \
  --epochs 20000 \
  --batch_size 10240
