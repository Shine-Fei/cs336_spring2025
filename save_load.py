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
    input_ = torch.from_numpy(x_batch[:, :-1]).long().to(device)
    target = torch.from_numpy(x_batch[:, 1:]).long().to(device)
    return input_, target

class seq_dataloader:
    def __init__(self, data, batch_size, context_length, device):
        self.data = data
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.indices = np.arange(0, len(data) - context_length)
        self.p = 0
        self.shuffle()
    
    def shuffle(self):
        np.random.shuffle(self.indices)
        self.p = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.p + self.batch_size > len(self.indices):
            self.shuffle()
        idx_batch = self.indices[self.p : self.p + self.batch_size]
        self.p += self.batch_size
        x_batch = np.stack([self.data[i: i + self.context_length + 1] for i in idx_batch])
        input_ = torch.from_numpy(x_batch[:, :-1]).long().to(self.device)
        target = torch.from_numpy(x_batch[:, 1:]).long().to(self.device)
        return input_, target

class val_dataloader:
    def __init__(self, data, batch_size, context_length, device):
        self.data = data
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.indices = np.arange(0, len(data) - context_length)
        self.p = 0
        
    def __iter__(self):
        self.p = 0
        return self
    
    def __next__(self):
        if self.p >= len(self.indices):
            raise StopIteration
        end = min(self.p + self.batch_size, len(self.indices))
        idx_batch = self.indices[self.p : end]
        self.p = end
        x_batch = np.stack([self.data[i: i + self.context_length + 1] for i in idx_batch])
        input_ = torch.from_numpy(x_batch[:, :-1]).long().to(self.device)
        target = torch.from_numpy(x_batch[:, 1:]).long().to(self.device)
        return input_, target

def sample_val_batches(data, batch_size, context_length, n_batches, device):
    indices = np.random.choice(len(data) - context_length, size=batch_size * n_batches, replace=False)
    batches = []

    for i in range(n_batches):
        idx_batch = indices[i * batch_size : (i + 1) * batch_size]
        x_batch = np.stack([data[i : i + context_length + 1] for i in idx_batch])
        input_ = torch.from_numpy(x_batch[:, :-1]).long().to(device)
        target = torch.from_numpy(x_batch[:, 1:]).long().to(device)
        batches.append((input_, target))
    
    return batches


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