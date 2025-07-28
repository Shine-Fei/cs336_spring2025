import torch
import numpy as np
import argparse
import logging
import wandb
from save_load import *
from my_module import adamw, transformer_lm, cross_entropy, lr_cosine_schedule, update_lr


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
    parser.add_argument('--lr_max',type=float,default=1e-5)
    parser.add_argument('--lr_min',type=float,default=1e-3)
    parser.add_argument('--t_w',type=int,default=100)
    parser.add_argument('--t_c',type=int,default=15000)

    parser.add_argument('--weight_decay',type=float,default=0.01)
    parser.add_argument('--eps',type=float,default=1e-8)
    parser.add_argument('--betas',type=tuple,default=(0.9, 0.999))
    parser.add_argument('--device',type=str,default='mps')
    parser.add_argument('--data_path',type=str,required=True)
    parser.add_argument('--save_path',type=str,default='result/state.pt')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--log_interval',type=int,default=100)
    parser.add_argument('--train_mode',type=str,default=None)

    args = parser.parse_args()

    device = torch.device(args.device)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    if args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))

    
    data = np.memmap(args.data_path, mode='r')
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
            if iteration % 1000 == 0:
                save_checkpoint(model, optimizer, iteration, args.save_path)
            
            iteration += 1
    
    save_checkpoint(model, optimizer, iteration, args.save_path)

if __name__ == '__main__':
    main()
