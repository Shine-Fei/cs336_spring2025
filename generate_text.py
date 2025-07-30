import torch
import numpy as np
import json
import base64
import os
import argparse
from my_module import transformer_lm
from my_tokenizer import bpe_tokenizer


if torch.cuda.is_available():
        device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default='result/tokenizer/bpe_vocab.json')
    parser.add_argument('--merges_path', type=str, default='result/tokenizer/bpe_merges.txt')
    parser.add_argument('--model_path', type=str, default='result/state_best_1.62.pt')
    parser.add_argument('--max_token', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p_value', type=float, default=0.9)
    parser.add_argument('--end_token', type=str, default='<|endoftext|>')
    parser.add_argument('--num_processes',type=int,default=10)
    parser.add_argument('--context_length',type=int,default=256)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--d_model',type=int,default=512)
    parser.add_argument('--vocab_size',type=int,default=10000)
    parser.add_argument('--rope_theta',type=int,default=10000)
    parser.add_argument('--num_layers',type=int,default=4)
    parser.add_argument('--num_heads',type=int,default=16)
    parser.add_argument('--d_ff',type=int,default=1344)
    parser.add_argument('--total_tokens',type=int,default=40960000)

    args = parser.parse_args()

    d_k = args.d_model // args.num_heads
    model = transformer_lm(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta, d_k, device=device)
    state = torch.load(args.model_path)
    model.load_state_dict(state['model'])
    model.eval()

    tokenizer = bpe_tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"], num_processes=args.num_processes)
    end_byte = args.end_token.encode('utf-8')
    if end_byte in tokenizer.vocab_inv:
        end_id = tokenizer.vocab_inv[end_byte]
        print('Generating with end token... \n')
    else:
        print('Generating without end token... \n')
    prompt = input('type your input:')
    prompt_id = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_id], dtype=torch.long, device=device)
    output_id = model.decoding(prompt_tensor, max_token=args.max_token, temperature=args.temperature, top_p_value=args.top_p_value, end_id=end_id)
    output_id = output_id[0].cpu().tolist()
    total_id = prompt_id + output_id
    reply = tokenizer.decode(total_id)
    print('\n'+reply)

if __name__ == '__main__':
    main()