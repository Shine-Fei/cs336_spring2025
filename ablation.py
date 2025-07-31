import torch
import numpy as np
import argparse
import json
import base64
import logging
import wandb
from tqdm import tqdm
import os
from save_load import *
from my_module import adamw, transformer_noRMSNorm, transformer_postnorm, transformer_nope, transformer_silu, cross_entropy, lr_cosine_schedule, update_lr, gradient_clipping
from train_tokenizer import run_train_bpe
from my_tokenizer import bpe_tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--context_length',type=int,default=256)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--d_model',type=int,default=512)
    parser.add_argument('--vocab_size',type=int,default=10000)
    parser.add_argument('--rope_theta',type=int,default=10000)
    parser.add_argument('--num_layers',type=int,default=4)
    parser.add_argument('--num_heads',type=int,default=16)
    parser.add_argument('--d_ff',type=int,default=1344)
    parser.add_argument('--total_tokens',type=int,default=40960000)
    parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--lr_max',type=float,default=1e-3)
    parser.add_argument('--lr_min',type=float,default=1e-5)
    parser.add_argument('--t_w',type=int,default=500)
    parser.add_argument('--t_c',type=int,default=5000)
    parser.add_argument('--weight_decay',type=float,default=0.01)
    parser.add_argument('--eps',type=float,default=1e-8)
    parser.add_argument('--betas',type=tuple,default=(0.9, 0.999))
    parser.add_argument('--valid_batch',type=int,default=50)
    parser.add_argument('--train_data_path',type=str,required=True)
    parser.add_argument('--train_cache_path', type=str, default='result/tokenizer/train_data.bin')
    parser.add_argument('--valid_data_path',type=str,required=True)
    parser.add_argument('--valid_cache_path', type=str, default='result/tokenizer/valid_data.bin')
    parser.add_argument('--vocab_path', type=str, default='result/tokenizer/bpe_vocab.json')
    parser.add_argument('--merges_path', type=str, default='result/tokenizer/bpe_merges.txt')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--log_interval',type=int,default=100)
    parser.add_argument('--save_interval',type=int,default=1000)
    parser.add_argument('--train_mode',type=str,default=None)
    parser.add_argument('--encode_mode',type=str,default='iter')
    parser.add_argument('--num_processes',type=int,default=10)
    parser.add_argument('--max_norm',type=float,default=1.0)

    parser.add_argument(
        '--ablation',
        type=str.lower,
        required=True,
        choices=['rmsnorm', 'post_norm', 'nope', 'silu'],
        help='Ablation type: choose from [rmsnorm, post_norm, nope, silu]'
    )
    parser.add_argument('--save_path',type=str)
    parser.add_argument('--best_path',type=str)

    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = f'result/ablation/{args.ablation}/state.pt'
    if args.best_path is None:
        args.best_path = f'result/ablation/{args.ablation}/state_best.pt'

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
        tokens, merges = run_train_bpe(input_path=args.train_data_path, 
                                       vocab_size=args.vocab_size, special_tokens=["<|endoftext|>"],num_processes=16)
        # save vocab.json
        vocab_b64 = {k: base64.b64encode(v).decode('ascii') for k, v in tokens.items()}
        with open(args.vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_b64, f, ensure_ascii=False)

        # save merges.txt
        with open(args.merges_path, "w", encoding="utf-8") as f:
            for a, b in merges:
                a_str = base64.b64encode(a).decode("utf-8")
                b_str = base64.b64encode(b).decode("utf-8")
                f.write(f"{a_str} {b_str}\n")
        logging.info(f"Tokenizer saved to {args.vocab_path} and {args.merges_path}")
        tokenizer = bpe_tokenizer(tokens, merges, special_tokens=["<|endoftext|>"], num_processes=args.num_processes)
    else:
        logging.info("Loading existing BPE tokenizer...")
        tokenizer = bpe_tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"], num_processes=args.num_processes)

    if not os.path.exists(args.train_cache_path):
        os.makedirs(os.path.dirname(args.train_cache_path), exist_ok=True)
        logging.info("Encoding training text and saving to cache...")
        if args.encode_mode == 'iter':
            with open(args.train_data_path, "r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)
            with open(args.train_data_path, 'r', encoding='utf-8') as f:
                encoded = np.fromiter(tokenizer.encode_iterable(f, total=total_lines), dtype=np.uint16)
        else:
            with open(args.train_data_path, "r", encoding="utf-8") as f:
                corpus_contents = f.read()
            encoded = tokenizer.encode(corpus_contents)
        encoded.tofile(args.train_cache_path)

    if not os.path.exists(args.valid_cache_path):
        os.makedirs(os.path.dirname(args.valid_cache_path), exist_ok=True)
        logging.info("Encoding validation text and saving to cache...")
        if args.encode_mode == 'iter':
            with open(args.valid_data_path, "r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)
            with open(args.valid_data_path, 'r', encoding='utf-8') as f:
                encoded = np.fromiter(tokenizer.encode_iterable(f, total=total_lines), dtype=np.uint16)
        else:
            with open(args.valid_data_path, "r", encoding="utf-8") as f:
                corpus_contents = f.read()
            encoded = tokenizer.encode(corpus_contents)
        encoded.tofile(args.valid_cache_path)
        
    train_data = np.memmap(args.train_cache_path, mode='r', dtype=np.uint16)
    valid_data = np.memmap(args.valid_cache_path, mode='r', dtype=np.uint16)

    d_k = args.d_model // args.num_heads
    if args.ablation == 'rmsnorm':
        model = transformer_noRMSNorm(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta, d_k, device=device)
    elif args.ablation == 'post_norm':
        model = transformer_postnorm(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta, d_k, device=device)
    elif args.ablation == 'nope':
        model = transformer_nope(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta, d_k, device=device)
    
    elif args.ablation == 'silu':
        model = transformer_silu(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta, d_k, device=device)
    else:
        raise('invalid ablation parameters')

    optimizer = adamw(model.parameters(), lr=args.lr, weight_decay = args.weight_decay, betas=args.betas, eps=args.eps)

    if args.train_mode == 'continue':
        try:
            iteration = load_checkpoint(args.save_path, model, optimizer)
            print(f"Resumed from iteration {iteration}")
        except:
            raise('missing saving path or saved model')
    else:
        iteration = 0

    dataloader_train = seq_dataloader(train_data, args.batch_size, args.context_length, device)
    #dataloader_valid = val_dataloader(valid_data, args.batch_size, args.context_length, device) #too slow for validating all data
    step_count = args.total_tokens // (args.batch_size * args.context_length)
    min_loss = float('inf')
    logging.info(f"Begin training, ablation for {args.ablation}")
    for epoch in range(args.epochs):
        #for _ in range(step_count):
        for x, y in dataloader_train:
            model.train()
            new_lr = lr_cosine_schedule(iteration, alpha_max=args.lr_max, alpha_min=args.lr_min, t_w= args.t_w, t_c= args.t_c)
            update_lr(optimizer, new_lr)
            #x, y = dataloader(data, args.batch_size, args.context_length, device)
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            gradient_clipping(model.parameters(), args.max_norm)
            optimizer.step()
            optimizer.zero_grad()

            #evaluate
            if iteration % args.log_interval == 0:
                model.eval()
                total_val_loss = 0
                count = 0
                with torch.no_grad():
                    #for x, y in tqdm(dataloader_valid, desc=f"Running validation for iter {iteration}"): #too slow for validating all data
                    for x, y in sample_val_batches(valid_data, args.batch_size, args.context_length, args.valid_batch, device):
                        valid_logits = model(x)
                        val_loss = cross_entropy(valid_logits, y)
                        total_val_loss += val_loss.item()
                        count += 1
                total_val_loss = total_val_loss / count
                logging.info(f"Iteration {iteration}: train_loss = {loss.item():.4f},  valid_loss = {total_val_loss:.4f}")

                if total_val_loss < min_loss:
                    min_loss = total_val_loss
                    save_checkpoint(model, optimizer, iteration, args.best_path)
                    logging.info(f"Iteration {iteration}: best model saved, valid_loss = {min_loss:.4f}")

                if args.wandb_project:
                    wandb.log({'train_loss': loss.item(), 'valid_loss':total_val_loss,'iteration': iteration, 'lr':new_lr})

            if iteration % args.save_interval == 0:
                save_checkpoint(model, optimizer, iteration, args.save_path)
                logging.info(f"Iteration {iteration}: model saved")
            
            iteration += 1

            if iteration % step_count == 0:
                break

    save_checkpoint(model, optimizer, iteration, args.save_path)

if __name__ == '__main__':
    main()
