import torch
import numpy as np
from typing import List, Tuple

class ReadoutNetwork(torch.nn.Module):

    def __init__(self, embedding_size: int = 128, hidden_size: int = 128, action_dims: List[int] = [6,10,10]) -> None: 
        
        super(ReadoutNetwork, self).__init__()

        self.__rho = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, np.prod(action_dims)),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:

        return self.__rho(h)


class BackupNetwork(torch.nn.Module):

    def __init__(self, embedding_size: int = 128, hidden_size: int = 128, action_dims: List[int] = [6,10,10]) -> None:
        
        super(BackupNetwork, self).__init__()

        self.__f = torch.nn.Sequential(
            torch.nn.Linear(2*embedding_size + 1 + np.prod(action_dims), hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, embedding_size)
        )
        self.__g = torch.nn.Sequential(
            torch.nn.Linear(2*embedding_size + 1 + np.prod(action_dims), hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, embedding_size),
            torch.nn.Sigmoid()
        )

    def forward(self, h: torch.Tensor, h_prime: torch.Tensor, reward: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        
        x = torch.cat((h,h_prime,reward,probs), dim=1)

        return self.__f(x)*self.__g(x) + h

class ResidualConvolutionalNetwork(torch.nn.Module):

    def __init__(self, n_input_maps: int, n_residual_blocks: int, channel_sizes: List[int], kernels: List[int], strides: List[int]):

        super(ResidualConvolutionalNetwork, self).__init__()

        io = []
        ks = []
        ss = []
        for _ in range(n_residual_blocks):
            io += channel_sizes
            ks += kernels
            ss += strides
        io = [n_input_maps] + io
        self.__layers = torch.nn.ModuleList([ torch.nn.Conv2d(i, o, k, s, 'same') for i,o,k,s in zip(io[:-1], io[1:], ks, ss)])
        self.__activations = torch.nn.ModuleList([ torch.nn.ReLU() for _ in ks])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, (layer, activation) in enumerate(zip(self.__layers, self.__activations)):
            x = activation(layer(x))
            if idx % 4 == 0:
                res = x.clone()
            if idx % 4 == 2:
                x = x + res
        
        return torch.flatten(x, start_dim=1)

class EmbeddingNetwork(ResidualConvolutionalNetwork):

    def __init__(self, state_dims: List[int] = [2,10,10],
                       n_residual_blocks: int = 3,
                       channel_sizes: List[int] = [64,64,64,32],
                       kernels: List[int] = [3,3,3,1],
                       strides: List[int] = [1,1,1,1],
                       embedding_size: int = 128,
                       hidden_size: int = 128):

        super().__init__(n_input_maps = state_dims[0], 
                         n_residual_blocks = n_residual_blocks, 
                         channel_sizes = channel_sizes,
                         kernels = kernels,
                         strides = strides)
        
        self.__epsilon = torch.nn.Sequential(
            torch.nn.Linear(state_dims[1]*state_dims[2]*channel_sizes[-1], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:

        s = super().forward(s)

        return self.__epsilon(s)

class PolicyNetwork(ResidualConvolutionalNetwork):

    def __init__(self, action_dims: List[int] = [6,10,10],
                       n_residual_blocks: int = 2,
                       channel_sizes: List[int] = [32,32,32,32],
                       kernels: List[int] = [3,3,3,1],
                       strides: List[int] = [1,1,1,1],
                       embedding_size: int = 128,
                       hidden_size: int = 128,
                       device = 'cpu'):

        super().__init__(n_input_maps = 1,
                         n_residual_blocks = n_residual_blocks, 
                         channel_sizes = channel_sizes,
                         kernels = kernels,
                         strides = strides)

        self.__psi = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, np.prod(action_dims))
        )

        self.__u = torch.nn.Sequential(
            torch.nn.Linear(2*embedding_size*channel_sizes[-1], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )

        self.__pi = torch.nn.Softmax(dim=1)

        self.__device = device

    def forward(self, h: torch.Tensor, h_primes: List[Tuple[int,torch.Tensor]], w_0: float = 0.5, w_1: float = 0.5) -> torch.Tensor:
        
        psi = self.__psi(h)
        new_dim = int(np.sqrt(2*h.size(dim=1)))
        x = []
        for _, h_prime in h_primes:
            x.append(torch.reshape(torch.cat((h, h_prime), dim=1), (-1,new_dim,new_dim)))
        x = torch.unsqueeze(torch.cat(x, dim=0), 1)
        x = self.__u(super().forward(x))
        u = []
        last_id = 0
        for idx, (child_id, _) in enumerate(h_primes):
            u.append(torch.zeros(1, child_id-last_id, device=self.__device))
            u.append(torch.unsqueeze(x[idx], 0))
            last_id = child_id+1
        u.append(torch.zeros(1, psi.size(1)-last_id, device=self.__device))
        u = torch.cat(u, dim=1)
        
        return  self.__pi(w_0*psi + w_1*u)
