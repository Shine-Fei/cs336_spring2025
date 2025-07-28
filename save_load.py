import torch
from einops import rearrange, einsum
import numpy as np

def dataloader(x, batch_size, context_length, device):
    '''
    x: numpy array (integer array with token IDs)
    batch_size: int
    context_length: int
    device: str

    return : 
        sampled input sequences: Tensor (batch_size, context_length)
        corresponding next-token targets : Tensor (batch_size, context_length)
    '''

    idx = np.random.randint(0, len(x) - context_length, size=batch_size)
    x_batch = np.stack([x[i: i + context_length + 1] for i in idx])
    input_ = torch.from_numpy(x_batch[:, :-1]).to(device)
    target = torch.from_numpy(x_batch[:, 1:]).to(device)
    return input_, target

def save_checkpoint(model, optimizer, iteration, out):
    '''
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iteration: int
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    '''
    state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),'iteration':iteration}
    torch.save(state, out)

def load_checkpoint(src, model, optimizer):
    '''
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    '''
    state = torch.load(src)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    return state['iteration']