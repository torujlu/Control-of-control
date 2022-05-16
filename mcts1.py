import numpy as np
import torch
from typing import List
from networks import ReadoutNetwork

class MCTS1(torch.nn.Module):

    def __init__(self, action_dims: List[int] = [3,4,6],
                       device = 'cpu'):

        super(MCTS1, self).__init__()

        self.__readout = ReadoutNetwork(np.prod(action_dims), np.prod(action_dims), action_dims).to(device)

        self.__saved_log_probs = []
        self.__saved_entropies = []
        self.__rewards = []
        self.__losses = []
        self.__running_rewards = []
        self.__success_ratios = []
        self.__device = device

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

    def forward(self, tree, n_simuls = 10, c = 1.0, gamma = 0.9):
        
        for _ in range(n_simuls):
            
            current_node = tree.get_root()
            done = current_node.get_done()
            children = current_node.get_children()

            while children and not done:

                Q = current_node.get_Q()
                N = current_node.get_N()
                action_mask = current_node.get_action_mask()
                masked_Q = action_mask*Q + (action_mask - 1)*1e6
                masked_N = action_mask*N + (action_mask - 1)
                unexplored = np.where(masked_N==0)[0]
                if len(unexplored)>0:
                    action = np.random.choice(unexplored)
                else:
                    for action in range(len(N)):
                        if N[action] > 0:
                            masked_Q[action] += c*np.sqrt(np.log(np.sum(N))/N[action])
                    action = np.random.choice(np.where(masked_Q==np.max(masked_Q))[0])
                current_node = current_node.next_node(action)
                done = current_node.get_done()
                children = current_node.get_children()

                if not children or done:
                    N[action] += 1
                    current_node.set_N(N)
            
            
            if not done:
                R = current_node.roll_out(gamma)
                current_node.expand()
            else:
                R = current_node.get_reward()
            
            parent = current_node.get_parent()
            while parent != None:
                reward = parent.get_reward()
                action = current_node.get_action()
                R = reward + gamma*R
                Q = parent.get_Q()
                N = parent.get_N()
                Q[action] = Q[action] + (R-Q[action])/(N[action]+1)
                N[action] = N[action] + 1
                parent.set_Q(Q)
                parent.set_N(N)
                current_node = parent
                parent = current_node.get_parent()
                tree.get_env().revert()        
        
        current_node = tree.get_root()
        N = current_node.get_N()
        action_mask = current_node.get_action_mask()
        probs = self.__readout(torch.unsqueeze(torch.tensor(N).to(self.__device),0))

        return probs, action_mask