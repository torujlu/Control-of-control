import torch
import numpy as np
from typing import List
from networks import ReadoutNetwork, BackupNetwork, EmbeddingNetwork, PolicyNetwork
from torch.distributions.bernoulli import Bernoulli
class MCTSnet(torch.nn.Module):

    def __init__(self, embedding_size: int = 128,
                       readout_hidden_size: int = 128,
                       backup_hidden_size: int = 128,
                       action_dims: List[int] = [3,4,6],
                       state_dims: List[int] = [2,10,10],
                       embedding_n_residual_blocks: int = 3,
                       embedding_channel_sizes: List[int] = [64,64,64,32],
                       embedding_kernels: List[int] = [3,3,3,1],
                       embedding_strides: List[int] = [1,1,1,1],
                       policy_n_residual_blocks: int = 2,
                       policy_channel_sizes: List[int] = [32,32,32,16],
                       policy_kernels: List[int] = [3,3,3,1],
                       policy_strides: List[int] = [1,1,1,1],
                       policy_hidden_size: int = 128,
                       device = 'cpu'):

        super(MCTSnet, self).__init__()
        
        self.__readout = ReadoutNetwork(embedding_size, readout_hidden_size, action_dims).to(device)
        self.__backup = BackupNetwork(embedding_size, backup_hidden_size).to(device)
        self.__embedding = EmbeddingNetwork(state_dims, embedding_n_residual_blocks, embedding_channel_sizes,
                                            embedding_kernels, embedding_strides, embedding_size).to(device)
        self.__policy = PolicyNetwork(action_dims, policy_n_residual_blocks, policy_channel_sizes,
                                      policy_kernels, policy_strides, embedding_size, policy_hidden_size, device).to(device)

        self.__saved_log_probs = []
        self.__saved_entropies = []
        self.__rewards = []
        self.__losses = []
        self.__running_rewards = []
        self.__success_ratios = []
    
    def get_saved_log_probs(self):
        return self.__saved_log_probs
    
    def update_saved_log_probs(self, log_prob):
        self.__saved_log_probs.append(log_prob)
    
    def delete_saved_log_probs(self):
        del self.__saved_log_probs[:]
    
    def get_saved_entropies(self):
        return self.__saved_entropies
    
    def update_saved_entropies(self, entropy):
        self.__saved_entropies.append(entropy)
    
    def delete_saved_entropies(self):
        del self.__saved_entropies[:]
    
    def get_rewards(self):
        return self.__rewards
    
    def update_rewards(self, reward):
        self.__rewards.append(reward)

    def delete_rewards(self):
        del self.__rewards[:]
    
    def get_last_episode_loss(self):
        return self.__losses[-1]
    
    def update_losses(self, loss):
        self.__losses.append(loss)
    
    def get_losses(self):
        return self.__losses
    
    def update_running_rewards(self, running_reward):
        self.__running_rewards.append(running_reward)
    
    def get_running_rewards(self):
        return self.__running_rewards
    
    def update_success_ratios(self, success_ratio):
        self.__success_ratios.append(success_ratio)
    
    def get_success_ratios(self):
        return self.__success_ratios

    def forward(self, tree, internal_M, n_simuls = 10, gamma = 0.9, internal_control = False):
        
        simul_num = 0
        simul_done = False
        while not simul_done:
            simul_num += 1

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
            #R = 0
            while not(current_node.is_root()):
                h = parent.get_h()
                h_prime = current_node.get_h()
                reward = current_node.get_reward()
                action = current_node.get_action()
                #R = reward + gamma*R
                parent.set_h(self.__backup(h, h_prime, reward, action))
                #print(parent.get_h()-h)
                current_node = parent
                parent = current_node.get_parent()
                tree.get_env().revert()
            
            if internal_control:
                cont_prob = internal_M(tree.get_root().get_h().detach().clone())
                m = Bernoulli(cont_prob)
                cont = m.sample()
                if cont.item() < 1:
                    simul_done = True
                    internal_M.update_saved_log_probs(m.log_prob(cont))
                    internal_M.update_saved_n_simuls(simul_num)
            elif simul_num >= n_simuls:
                simul_done = True
        
        current_node = tree.get_root()
        h = current_node.get_h()
        action_mask = current_node.get_action_mask()
        probs = self.__readout(h)        

        return probs, action_mask
