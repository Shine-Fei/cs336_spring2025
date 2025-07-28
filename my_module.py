import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Float, Int
from torch import Tensor
import math
from collections.abc import Callable, Iterable
from typing import Optional

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def my_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    #in_features = in_features - in_features.max(dim=dim, keepdim=True).values
    in_features = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    exp_x = torch.exp(in_features)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
    rope = None,
    token_positions = None
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    if rope:
        Q = rope(Q, token_positions)
        K = rope(K, token_positions)
    pre_soft = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys') / math.sqrt(d_k)
    if mask is not None:
        pre_soft = pre_soft.masked_fill(~mask.bool(), -torch.inf)
    att = my_softmax(pre_soft, dim=-1)
    return einsum(att, V, '... queries keys, ... keys d_v -> ... queries d_v')

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    log_sum = inputs.logsumexp(dim=-1)
    tar_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = -tar_logits + log_sum
    return loss.mean()

def lr_cosine_schedule(t, alpha_max, alpha_min, t_w, t_c):
    if t < t_w:
        alpha_t = t / t_w * alpha_max
    elif t > t_c:
        alpha_t = alpha_min
    else:
        alpha_t = alpha_min + 0.5 * (1 + math.cos(math.pi * ((t - t_w) / (t_c - t_w)))) * (alpha_max - alpha_min)

    return alpha_t

def update_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def gradient_clipping(params, max_norm, eps = 1e-6):
    grads = [p.grad.detach() for p in params if p.grad is not None]
    #l2_nrom = torch.norm(torch.cat([g.view(-1) for g in grads]), p=2)
    l2_nrom = torch.norm(torch.stack([torch.norm(g, p=2) for g in grads]), p=2)
    if l2_nrom >= max_norm:
        scale = max_norm / (l2_nrom + eps)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)
    return params


class my_linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.kwargs = {'device':device, 'dtype':dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **self.kwargs))
        #self.bias = nn.Parameter(torch.zeros(out_features, **self.kwargs))
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_ (self.weight, mean = 0, std = std, a = -3*std, b = 3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T #no bias

class my_embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.kwargs = {'device':device, 'dtype':dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **self.kwargs))
        torch.nn.init.trunc_normal_ (self.weight, mean = 0, std = 1, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class my_RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.kwargs = {'device':device, 'dtype':dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **self.kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(self.eps + torch.mean(x**2, dim=-1, keepdim=True))
        result = (x / rms) * self.weight
        return result.to(in_dtype)

class my_swiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        '''
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        '''
        super().__init__()
        self.kwargs = {'device':device, 'dtype':dtype}
        self.w1 = my_linear(d_model, d_ff, **self.kwargs)
        self.w2 = my_linear(d_ff, d_model, **self.kwargs)
        self.w3 = my_linear(d_model, d_ff, **self.kwargs)

    def SiLU(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.SiLU(self.w1(x)) * self.w3(x))
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        '''
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        '''
        super().__init__()
        half_l = d_k // 2
        #k = torch.arange(half_l, dtype=torch.float32, device=device)
        k = torch.arange(half_l, device=device)
        inv_k = 1 / (theta ** (k / half_l))
        #i = torch.arange(max_seq_len, dtype=torch.float32,device=device)
        i = torch.arange(max_seq_len, device=device)
        theta_ik = einsum(i, inv_k, 'max_seq_len, half_l -> max_seq_len half_l')
        cos = torch.cos(theta_ik)
        sin = torch.sin(theta_ik)
        self.register_buffer('cos_cache', cos, persistent=False)
        self.register_buffer('sin_cache', sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        x(in_query_or_key) (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
        '''
        #in_dtype = x.dtype
        #x = x.to(torch.float32)
        x_pair = rearrange(x, '... sequence_length (d_pair two) -> ... sequence_length d_pair two', two = 2)
        cos = self.cos_cache[token_positions] #[batch_size, max_seq_len, d_k // 2]
        sin = self.sin_cache[token_positions]
        rot_m = torch.stack(
            (torch.stack((cos, -sin), dim=-1),
             torch.stack((sin, cos), dim=-1)
            ),
            dim=-2
        )
        x_rot = einsum(rot_m, x_pair, '... d_pair i j , ... d_pair j-> ... d_pair i')
        out = rearrange(x_rot, '... d_pair i -> ... (d_pair i)', i = 2)
        #return out.to(in_dtype)
        return out
    
class my_multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, rope=None):
        super().__init__()
        self.kwargs = {'device':device, 'dtype':dtype}
        self.rope = rope
        self.h = num_heads
        self.d_model = d_model
        self.qkv = my_linear(d_model, 3 * d_model, **self.kwargs) #堆叠h*dk，h*dk，h*dv
        self.output_proj = my_linear(d_model, d_model, **self.kwargs)

    def forward(self, x: torch.Tensor, token_positions = None) -> torch.Tensor:
        x_qkv = self.qkv(x) #( ... ,sequence_length ,3 * d_model)
        q, k, v = x_qkv.chunk(3, dim = -1)
        q = rearrange(q, '... len (h d_k) -> ... h len d_k', h = self.h)
        k = rearrange(k, '... len (h d_k) -> ... h len d_k', h = self.h)
        v = rearrange(v, '... len (h d_v) -> ... h len d_v', h = self.h)
        seq_len = v.shape[-2]
        mask =  torch.tril(torch.ones(seq_len,seq_len, **self.kwargs))
        mask_shape = [1] * (v.ndim - 2) + [seq_len, seq_len]
        mask = mask.view(*mask_shape)
        out = scaled_dot_product_attention(q, k, v, mask=mask, rope = self.rope, token_positions = token_positions) #(... h len d_v)
        out = rearrange(out, '... h len d_v -> ... len (h d_v)') #(... len d_model)
        return self.output_proj(out)
    
class multihead_self_attention_rope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.kwargs = {'device':device, 'dtype':dtype}
        self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, **self.kwargs)
        self.h = num_heads
        self.d_model = d_model
        self.qkv = my_linear(d_model, 3 * d_model, **self.kwargs) #堆叠h*dk，h*dk，h*dv
        self.output_proj = my_linear(d_model, d_model, **self.kwargs)

    def forward(self, x: torch.Tensor, token_positions = None) -> torch.Tensor:
        x_qkv = self.qkv(x) #( ... ,sequence_length ,3 * d_model)
        #q, k, v = torch.split(x_qkv, [self.d_model, self.d_model, self.d_model], dim=-1)
        q, k, v = x_qkv.chunk(3, dim = -1)
        q = rearrange(q, '... len (h d_k) -> ... h len d_k', h = self.h)
        k = rearrange(k, '... len (h d_k) -> ... h len d_k', h = self.h)
        v = rearrange(v, '... len (h d_v) -> ... h len d_v', h = self.h)
        seq_len = v.shape[-2]
        mask =  torch.tril(torch.ones(seq_len,seq_len, **self.kwargs))
        mask_shape = [1] * (v.ndim - 2) + [seq_len, seq_len]
        mask = mask.view(*mask_shape)
        out = scaled_dot_product_attention(q, k, v, mask=mask, rope = self.rope, token_positions = token_positions) #(... h len d_v)
        out = rearrange(out, '... h len d_v -> ... len (h d_v)') #(... len d_model)
        return self.output_proj(out)
    

class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.kwargs = {'device':device, 'dtype':dtype}
        self.attn = multihead_self_attention_rope(d_model, num_heads, theta, d_k, max_seq_len, **self.kwargs)
        self.ln1 = my_RMSNorm(d_model, **self.kwargs)
        self.ln2 = my_RMSNorm(d_model, **self.kwargs)
        self.ffn = my_swiglu(d_model, d_ff, **self.kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        自动拼接 q_proj + k_proj + v_proj -> self.qkv.weight
        并将 output_proj 的权重正常加载
        """
        q_key = next(k for k in state_dict if k.endswith('q_proj.weight'))
        prefix = q_key.rsplit('q_proj.weight', 1)[0]  # e.g., 'layers.0.attn.'

        q = state_dict.pop(prefix + 'q_proj.weight')
        k = state_dict.pop(prefix + 'k_proj.weight')
        v = state_dict.pop(prefix + 'v_proj.weight')

        # 拼接为 qkv
        qkv_weight = torch.cat([q, k, v], dim=0)
        state_dict[prefix + 'qkv.weight'] = qkv_weight
        return super().load_state_dict(state_dict, strict)
    
    def forward(self, x, token_positions: Int[Tensor, "batch seq_len"] | None = None):
        '''
        x: (Float[Tensor, "batch sequence_length d_model"]):
        '''
        if token_positions is None:
        #先构造token_positions
            batch, seq_len , _= x.shape
            #token_positions = torch.arange(seq_len, **self.kwargs)
            #token_positions = token_positions.unsqueeze(0).expand(batch, seq_len)
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch, 1)
        
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
    
class transformer_lm(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, theta: float, d_k: int, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert d_model == d_k * num_heads, "d_model must equal d_k * num_heads"
        self.kwargs = {'device':device, 'dtype':dtype}
        self.token_embeddings = my_embedding(vocab_size, d_model, **self.kwargs)
        self.layers = nn.ModuleList([transformer_block(d_model, num_heads, d_ff, theta, d_k, context_length, **self.kwargs) for _ in range(num_layers)])
        self.ln_final = my_RMSNorm(d_model, **self.kwargs)
        self.lm_head = my_linear(d_model, vocab_size, **self.kwargs)
    
    def load_state_dict(self, state_dict, strict: bool = True):
        """
        自动拼接 q_proj + k_proj + v_proj -> self.qkv.weight
        并将 output_proj 的权重正常加载
        """
        state_dict = state_dict.copy()  # 避免就地 pop
        for l in range(len(self.layers)):
            prefix = f'layers.{l}.attn.'
            q_key = prefix + 'q_proj.weight'
            k_key = prefix + 'k_proj.weight'
            v_key = prefix + 'v_proj.weight'
            qkv_key = prefix + 'qkv.weight'

            if q_key in state_dict and k_key in state_dict and v_key in state_dict:
                q = state_dict.pop(q_key)
                k = state_dict.pop(k_key)
                v = state_dict.pop(v_key)
                state_dict[qkv_key] = torch.cat([q, k, v], dim=0)

        return super().load_state_dict(state_dict, strict)
    
    def forward(self, input_indices: Int[Tensor, "batch seq_len"], token_positions: Int[Tensor, "batch seq_len"] | None = None):
        x = self.token_embeddings(input_indices)
        if token_positions is None:
        #先构造token_positions
            batch, seq_len , _= x.shape
            #token_positions = torch.arange(seq_len, **self.kwargs)
            #token_positions = token_positions.unsqueeze(0).expand(batch, seq_len)
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch, 1)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
    
class adamw(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, 'betas':betas, 'eps':eps, 'weight_decay':weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1, beta2 = group['betas']
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", 0)
                v = state.get("v", 0)
                grad = p.grad.data # Get the gradient of loss with respect to p.
                m = beta1 * m + (1-beta1) * grad
                v = beta2 * v + (1-beta2) * grad ** 2
                alpha_t = lr * math.sqrt(1-(beta2)**t) / (1 - (beta1)**t)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)# Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss
    
