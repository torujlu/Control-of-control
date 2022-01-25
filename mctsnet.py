import torch
import numpy as np
from typing import List
from networks import ReadoutNetwork, BackupNetwork, EmbeddingNetwork, PolicyNetwork

class MCTSnet(torch.nn.Module):

    def __init__(self, embedding_size: int = 128,
                       hidden_size: int = 128,
                       action_dims: List[int] = [6,10,10],
                       state_dims: List[int] = [2,10,10],
                       embedding_n_residual_blocks: int = 3,
                       embedding_channel_sizes: List[int] = [64,64,64,32],
                       embedding_kernels: List[int] = [3,3,3,1],
                       embedding_strides: List[int] = [1,1,1,1],
                       policy_n_residual_blocks: int = 2,
                       policy_channel_sizes: List[int] = [32,32,32,32],
                       policy_kernels: List[int] = [3,3,3,1],
                       policy_strides: List[int] = [1,1,1,1],
                       device = 'cpu'):

        super(MCTSnet, self).__init__()
        
        self.__readout = ReadoutNetwork(embedding_size, hidden_size, action_dims).to(device)
        self.__backup = BackupNetwork(embedding_size, hidden_size, action_dims).to(device)
        self.__embedding = EmbeddingNetwork(state_dims, embedding_n_residual_blocks, embedding_channel_sizes,
                                            embedding_kernels, embedding_strides, embedding_size, hidden_size).to(device)
        self.__policy = PolicyNetwork(action_dims, policy_n_residual_blocks, policy_channel_sizes,
                                      policy_kernels, policy_strides, embedding_size, hidden_size, device).to(device)

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

    def forward(self, tree, n_simuls = 10):

        for _ in range(n_simuls):

            current_node = tree.get_root()
            done = current_node.get_done()
            children = current_node.get_children()

            while children and not done:

                h = current_node.get_h()
                h_primes = []
                
                for (child_id, child) in sorted(children):
                    h_primes.append((child_id, child.get_h()))
                
                probs = self.__policy(h, h_primes)
                current_node = current_node.next_node(probs)
                done = current_node.get_done()
                children = current_node.get_children()
            
            state = current_node.get_state()
            h = self.__embedding(state)
            current_node.set_h(h)
            if not done:
                current_node.expand()

            parent = current_node.get_parent()
            while not(current_node.is_root()):
                h = parent.get_h()
                h_prime = current_node.get_h()
                reward = current_node.get_reward()
                probs = current_node.get_probs()
                parent.set_h(self.__backup(h, h_prime, reward, probs))
                current_node = parent
                parent = current_node.get_parent()
                tree.get_env().revert()
        
        current_node = tree.get_root()
        h = current_node.get_h()
        probs_mask = current_node.get_probs_mask()
        probs = self.__readout(h)        

        return probs, probs_mask
