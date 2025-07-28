import torch
import numpy as np
import argparse
import json
import base64
import logging
import wandb
import os
from save_load import *
from my_module import adamw, transformer_lm, cross_entropy, lr_cosine_schedule, update_lr
from train_tokenizer import run_train_bpe
from my_tokenizer import bpe_tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--context_length',type=int,default=256)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--d_model',type=int,default=512)
    parser.add_argument('--vocab_size',type=int,default=10000)
    parser.add_argument('--rope_theta',type=int,default=10000)
    parser.add_argument('--num_layers',type=int,default=4)
    parser.add_argument('--num_heads',type=int,default=16)
    parser.add_argument('--d_ff',type=int,default=1344)
    parser.add_argument('--total_tokens',type=int,default=327680000)
    parser.add_argument('--lr',type=float,default=1e-6)
    parser.add_argument('--lr_max',type=float,default=1e-4)
    parser.add_argument('--lr_min',type=float,default=1e-8)
    parser.add_argument('--t_w',type=int,default=100)
    parser.add_argument('--t_c',type=int,default=15000)
    parser.add_argument('--weight_decay',type=float,default=0.01)
    parser.add_argument('--eps',type=float,default=1e-8)
    parser.add_argument('--betas',type=tuple,default=(0.9, 0.999))
    parser.add_argument('--data_path',type=str,required=True)
    parser.add_argument('--save_path',type=str,default='result/state.pt')
    parser.add_argument('--vocab_path', type=str, default='result/tokenizer/bpe_vocab.json')
    parser.add_argument('--merges_path', type=str, default='result/tokenizer/bpe_merges.txt')
    parser.add_argument('--cache_path', type=str, default='result/tokenizer/train_data.bin')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--log_interval',type=int,default=100)
    parser.add_argument('--save_interval',type=int,default=1000)
    parser.add_argument('--train_mode',type=str,default=None)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    if args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))

    if not os.path.exists(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if not (os.path.exists(args.vocab_path) and os.path.exists(args.merges_path)):
        os.makedirs(os.path.dirname(args.vocab_path), exist_ok=True)
        os.makedirs(os.path.dirname(args.merges_path), exist_ok=True)
        logging.info("Training BPE tokenizer...")
        
        # use ByteLevelBPETokenizer to train tokenizer
        tokens, merges = run_train_bpe(input_path=args.data_path, 
                                       vocab_size=args.vocab_size, special_tokens=["<|endoftext|>"],num_processes=16)
        # save vocab.json 和 merges.txt
        vocab_b64 = {k: base64.b64encode(v).decode('ascii') for k, v in tokens.items()}
        with open(args.vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_b64, f, ensure_ascii=False)

        # save merges
        with open(args.merges_path, "w", encoding="utf-8") as f:
            for a, b in merges:
                a_str = base64.b64encode(a).decode("utf-8")
                b_str = base64.b64encode(b).decode("utf-8")
                f.write(f"{a_str} {b_str}\n")
        logging.info(f"Tokenizer saved to {args.vocab_path} and {args.merges_path}")
        tokenizer = bpe_tokenizer(tokens, merges, special_tokens=["<|endoftext|>"])
    else:
        logging.info("Loading existing BPE tokenizer...")
        tokenizer = bpe_tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"])

    if not os.path.exists(args.cache_path):
        os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)
        logging.info("Encoding text and saving to cache...")
        with open(args.data_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        with open(args.data_path, 'r', encoding='utf-8') as f:
            encoded = np.fromiter(tokenizer.encode_iterable(f, total=total_lines), dtype=np.uint16)
        encoded.tofile(args.cache_path)
        
    data = np.memmap(args.cache_path, mode='r', dtype=np.uint16)

    d_k = args.d_model // args.num_heads
    model = transformer_lm(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta, d_k, device=device)
    optimizer = adamw(model.parameters(), lr=args.lr, weight_decay = args.weight_decay, betas=args.betas, eps=args.eps)

    if args.train_mode == 'continue':
        try:
            iteration = load_checkpoint(args.save_path, model, optimizer)
            print(f"Resumed from iteration {iteration}")
        except:
            raise('missing saving path or saved model')
    else:
        iteration = 0
    model.train()

    step_count = args.total_tokens // (args.batch_size * args.context_length)
    for epoch in range(args.epochs):
        for _ in range(step_count):
            new_lr = lr_cosine_schedule(iteration, alpha_max=args.lr_max, alpha_min=args.lr_min, t_w= args.t_w, t_c= args.t_c)
            update_lr(optimizer, new_lr)
            x, y = dataloader(data, args.batch_size, args.context_length, device)
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if iteration % args.log_interval == 0:
                logging.info(f"Iteration {iteration}: train_loss = {loss.item():.4f}")
                if args.wandb_project:
                    wandb.log({'loss': loss.item(), 'iteration': iteration, 'lr':new_lr})
            if iteration % args.save_interval == 0:
                save_checkpoint(model, optimizer, iteration, args.save_path)
                logging.info(f"Iteration {iteration}: model saved")
            
            iteration += 1

    save_checkpoint(model, optimizer, iteration, args.save_path)

if __name__ == '__main__':
    main()
