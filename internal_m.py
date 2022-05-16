import torch
from typing import List

class InternalM(torch.nn.Module):

    def __init__(self, embedding_size: int = 128, hidden_size: int = 128) -> None: 

        super(InternalM, self).__init__()

        self.__internal_M = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )

        self.__saved_log_probs = []
        self.__archived_log_probs = []
        self.__saved_n_simuls = []
        self.__archived_n_simuls = []
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
    
    def get_saved_n_simuls(self):
        return self.__saved_n_simuls
    
    def update_saved_n_simuls(self, n_simuls):
        self.__saved_n_simuls.append(n_simuls)
    
    def delete_saved_n_simuls(self):
        self.__archived_n_simuls.append(self.__saved_n_simuls)
        del self.__saved_n_simuls[:]
    
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

    def forward(self, h: torch.Tensor) -> torch.Tensor:

        return self.__internal_M(h)