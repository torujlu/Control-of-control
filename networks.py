import torch
import numpy as np
from typing import List

class ReadoutNetwork(torch.nn.Module):

    def __init__(self, embedding_size: int = 128, hidden_size: int = 128, action_dims: List[int] = [6,10,10]) -> None: 
        
        super(ReadoutNetwork, self).__init__()
        
        self.__input_layer = torch.nn.Linear(embedding_size, hidden_size)
        """
        self.__readout_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, action_dims[0]),
                                                     torch.nn.Linear(hidden_size, action_dims[1]),
                                                     torch.nn.Linear(hidden_size, action_dims[2])])
        """
        self.__readout_layer = torch.nn.Linear(hidden_size, np.prod(action_dims)) #np.sum(action_dims)
        self.__input_activation = torch.nn.ReLU()
        self.__readout_activation = torch.nn.Softmax(dim=1)
        #self.__readout_activations = torch.nn.ModuleList([torch.nn.Softmax(dim=1), torch.nn.Softmax(dim=1), torch.nn.Softmax(dim=1)])
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:

        x = self.__input_activation(self.__input_layer(h))
        """
        x1 = self.__readout_activations[0](self.__readout_layers[0](x))
        x2 = self.__readout_activations[1](self.__readout_layers[1](x))
        x3 = self.__readout_activations[2](self.__readout_layers[2](x))
        """
        return self.__readout_activation(self.__readout_layer(x)) #torch.cat((x1,x2,x3), dim=1)


class BackupNetwork(torch.nn.Module):

    def __init__(self, embedding_size: int = 128, action_dims: List[int] = [6,10,10]) -> None:
        
        super(BackupNetwork, self).__init__()

        self.__f_layer = torch.nn.Linear(2*embedding_size + 1 + np.prod(action_dims), embedding_size)
        self.__g_layer = torch.nn.Linear(2*embedding_size + 1 + np.prod(action_dims), embedding_size)

        self.__f_activation = torch.nn.ReLU()
        self.__g_activation = torch.nn.Sigmoid()

    def forward(self, h: torch.Tensor, h_prime: torch.Tensor, reward: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        
        x = torch.cat((h,h_prime,reward,probs), dim=1)

        xf = self.__f_activation(self.__f_layer(x))
        xg = self.__g_activation(self.__g_layer(x))

        return xf*xg + h

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
                       embedding_size: int = 128):

        super().__init__(n_input_maps = state_dims[0], 
                         n_residual_blocks = n_residual_blocks, 
                         channel_sizes = channel_sizes,
                         kernels = kernels,
                         strides = strides)
        
        self.__readout_layer = torch.nn.Linear(state_dims[1]*state_dims[2]*channel_sizes[-1], embedding_size)
        self.__readout_activation = torch.nn.ReLU()

    def forward(self, s: torch.Tensor) -> torch.Tensor:

        s = super().forward(s)

        return self.__readout_activation(self.__readout_layer(s))

class PolicyNetwork(ResidualConvolutionalNetwork):

    def __init__(self, action_dims: List[int] = [6,10,10],
                       n_residual_blocks: int = 2,
                       channel_sizes: List[int] = [32,32,32,32],
                       kernels: List[int] = [3,3,3,1],
                       strides: List[int] = [1,1,1,1],
                       embedding_size: int = 128):

        super().__init__(n_input_maps = 1,
                         n_residual_blocks = n_residual_blocks, 
                         channel_sizes = channel_sizes,
                         kernels = kernels,
                         strides = strides)
        """
        self.__prior_layers = torch.nn.ModuleList([torch.nn.Linear(2*embedding_size*channel_sizes[-1], action_dims[0]),
                                                   torch.nn.Linear(2*embedding_size*channel_sizes[-1], action_dims[1]),
                                                   torch.nn.Linear(2*embedding_size*channel_sizes[-1], action_dims[2])])
        self.__mlp_layers = torch.nn.ModuleList([torch.nn.Linear(embedding_size, action_dims[0]),
                                                 torch.nn.Linear(embedding_size, action_dims[1]),
                                                 torch.nn.Linear(embedding_size, action_dims[2])])

        self.__prior_activations = torch.nn.ModuleList([torch.nn.Softmax(dim=1), torch.nn.Softmax(dim=1), torch.nn.Softmax(dim=1)])
        self.__mlp_activations = torch.nn.ModuleList([torch.nn.Softmax(dim=1), torch.nn.Softmax(dim=1), torch.nn.Softmax(dim=1)])
        """
        self.__prior_layer = torch.nn.Linear(2*embedding_size*channel_sizes[-1], np.prod(action_dims)) #np.sum(action_dims)
        self.__mlp_layer = torch.nn.Linear(embedding_size, np.prod(action_dims)) #np.sum(action_dims)
        self.__prior_activation = torch.nn.Softmax(dim=1)
        self.__mlp_activation = torch.nn.Softmax(dim=1)

    def forward(self, h: torch.Tensor, h_primes: List[torch.Tensor], w_0: int = 0.5, w_1: int = 0.5) -> torch.Tensor:

        new_dim = int(np.sqrt(2*h.size(dim=1)))
        """
        x1 = self.__mlp_activations[0](self.__mlp_layers[0](h))
        x2 = self.__mlp_activations[1](self.__mlp_layers[1](h))
        x3 = self.__mlp_activations[2](self.__mlp_layers[2](h))

        x = torch.cat((x1,x2,x3), dim=1)
        """
        x = self.__mlp_activation(self.__mlp_layer(h))

        y = []
        for h_prime in h_primes:
            y.append(torch.reshape(torch.cat((h, h_prime), dim=1), (-1,new_dim,new_dim)))
        y = torch.unsqueeze(torch.cat(y, dim=0), 1)

        y = super().forward(y)
        """
        y1 = self.__prior_activations[0](self.__prior_layers[0](y))
        y2 = self.__prior_activations[1](self.__prior_layers[1](y))
        y3 = self.__prior_activations[2](self.__prior_layers[2](y))

        y = torch.cat((y1,y2,y3), dim=1)
        """
        y = self.__prior_activation(self.__prior_layer(y))
        probs = (torch.mul(x, w_0) + torch.mul(y.mean(0), w_1))
        
        return  probs
