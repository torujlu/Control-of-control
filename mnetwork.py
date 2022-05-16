import torch
from networks import ResidualConvolutionalNetwork
from typing import List

class MNetwork(ResidualConvolutionalNetwork):

    def __init__(self, state_dims: List[int] = [2,10,10],
                       n_residual_blocks: int = 2,
                       channel_sizes: List[int] = [32,32,32,16],
                       kernels: List[int] = [3,3,3,1],
                       strides: List[int] = [1,1,1,1],
                       hidden_size: int = 128,
                       max_M: int = 100):

        super().__init__(n_input_maps = state_dims[0], 
                         n_residual_blocks = n_residual_blocks, 
                         channel_sizes = channel_sizes,
                         kernels = kernels,
                         strides = strides)
        
        self.__M = torch.nn.Sequential(
            torch.nn.Linear(state_dims[1]*state_dims[2]*channel_sizes[-1], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, max_M),
            torch.nn.Softmax(dim=1)
        )

        self.__saved_log_probs = []
        self.__archived_log_probs = []
        self.__rewards = []
        self.__archived_rewards = []
        self.__losses = []
    
    def get_saved_log_probs(self):
        return self.__saved_log_probs
    
    def update_saved_log_probs(self, log_prob):
        self.__saved_log_probs.append(log_prob)
    
    def delete_saved_log_probs(self):
        self.__archived_log_probs.append(self.__saved_log_probs)
        del self.__saved_log_probs[:]
    
    def get_rewards(self):
        return self.__rewards
    
    def update_rewards(self, reward):
        self.__rewards.append(reward)

    def delete_rewards(self):
        self.__archived_rewards.append(self.__rewards)
        del self.__rewards[:]

    def get_last_episode_loss(self):
        return self.__losses[-1]
    
    def update_losses(self, loss):
        self.__losses.append(loss)

    def forward(self, s: torch.Tensor) -> torch.Tensor:

        s = super().forward(s)

        return self.__M(s)