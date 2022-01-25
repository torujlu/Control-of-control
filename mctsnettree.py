import torch
import gc
import numpy as np
from torch.distributions import Categorical

class MCTSnetTree():
	
    def __init__(self, env, embedding_size, device = 'cpu'):

        self.__env = env
        self.__embedding_size = embedding_size
        self.__device = device
        self.__root = MCTSnetNode(None, self, True)
        env_state = env.get_state()
        state = torch.unsqueeze(torch.tensor(env_state[:2]), 0).to(device)
        reward = torch.zeros(1,1).to(device)
        self.__root.set_state(state)
        self.__root.set_reward(reward)
        self.__root.set_done(False)
        self.__root.set_probs_mask(env.get_mask())

    def get_root(self):
        return self.__root
        
    def set_root(self, root):
        self.__root.make_root(False)
        root.make_root(True)
        self.__root = root

    def get_env(self):
        return self.__env
    
    def get_embedding_size(self):
        return self.__embedding_size
    
    def get_device(self):
        return self.__device
    
    def delete(self):
        self.__root.delete()
        torch.cuda.empty_cache()
        gc.collect()
		
class MCTSnetNode():
	
    def __init__(self, parent, tree, is_root):

        self.__device = tree.get_device()
        self.__parent = parent
        self.__tree = tree
        self.__is_root = is_root
        self.__children = []
        self.__h = torch.randn(1, tree.get_embedding_size()).to(self.__device)
    
    def get_state(self):
        return self.__state
    
    def set_state(self, state):
        self.__state = state
    
    def get_h(self):
        return self.__h

    def set_h(self, h):
        self.__h = h
    
    def get_reward(self):
        return self.__reward
    
    def set_reward(self, reward):
        self.__reward = reward

    def set_probs(self, probs):
        self.__probs = probs
    
    def get_probs(self):
        return self.__probs

    def get_done(self):
        return self.__done
    
    def set_done(self, done):
        self.__done = done

    def get_parent(self):
        return self.__parent
    
    def make_root(self, is_root):
        self.__is_root = is_root
    
    def is_root(self):
        return self.__is_root

    def expand(self):
        for action in range(len(self.__probs_mask)):
            if self.__probs_mask[action]:
                child = MCTSnetNode(self, self.__tree, False)
                self.__children.append((action, child))
    
    def get_children(self):
        return self.__children

    def get_probs_mask(self):
        return self.__probs_mask

    def set_probs_mask(self, probs_mask):
        self.__probs_mask = probs_mask
    
    def next_node(self, probs):
        probs = torch.squeeze(probs)
        n_grid, _, _ = self.__tree.get_env().get_dims()
        probs_mask = torch.tensor(self.__probs_mask).to(self.__device)
        masked_probs = probs*probs_mask
        if ~np.any(masked_probs.clone().cpu().detach().numpy()):
            masked_probs += probs_mask   
        masked_probs /= torch.sum(masked_probs)
        m = Categorical(masked_probs)
        action = m.sample().item()
        block = action//n_grid//n_grid
        y = (action - block*n_grid*n_grid)//n_grid
        x = action - block*n_grid*n_grid - y*n_grid
        env_action = np.array([block,y,x])

        env_state, env_reward, done = self.__tree.get_env().step(env_action)
        state = torch.unsqueeze(torch.tensor(env_state[:2]), 0).to(self.__device)
        reward = torch.unsqueeze(torch.tensor([env_reward]), 0).to(self.__device)
        for (child_id, child) in self.__children:
            if child_id == action:
                child.set_state(state)
                child.set_reward(reward)
                child.set_done(done)
                child.set_probs(torch.unsqueeze(probs, 0))
                child.set_probs_mask(self.__tree.get_env().get_mask())

                return child

    def delete(self):
        for (_, child) in self.__children:
            child.delete()
            del child
