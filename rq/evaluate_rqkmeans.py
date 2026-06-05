import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
import os

from rqkmeans_plus import RQVAE, apply_rqkmeans_plus_strategy, EmbDataset
from trainer import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(args):
    data = EmbDataset(args.data_path)
    logging.info(f"Loaded embeddings: {data.dim} dimensions, {len(data)} items")

    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=False,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )

    model = model.to(args.device)

    if args.pretrained_codebook_path:
        model = apply_rqkmeans_plus_strategy(model, args.pretrained_codebook_path, args.device)

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        logging.info(f"Loading checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        logging.warning("No checkpoint provided, evaluating untrained model")

    print(model)

    data_loader = DataLoader(data, num_workers=0,
                             batch_size=args.batch_size, shuffle=False,
                             pin_memory=True)

    trainer = Trainer(args, model, len(data_loader))

    loss, collision_rate = trainer.evaluate(data_loader)

    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Loss: {loss}")
    print(f"Collision Rate: {collision_rate}")
    print(f"{'='*60}")

    return loss, collision_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RQ-KMeans+ Model")

    parser.add_argument('--data_path', type=str,
                        default='../data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy')
    parser.add_argument('--pretrained_codebook_path', type=str,
                        default='../data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.codebooks_constrained.npz')
    parser.add_argument('--checkpoint_path', type=str, default="Apr-24-2026_17-44-20/best_collision_model.pth",
                        help='Path to model checkpoint (.pth file)')

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256])
    parser.add_argument('--e_dim', type=int, default=896)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--kmeans_iters', type=int, default=20)
    parser.add_argument('--sk_epsilons', type=float, default=1e-5)
    parser.add_argument('--sk_iters', type=int, default=100)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--log_interval', type=int, default=100)

    args = parser.parse_args()

    evaluate_model(args)
