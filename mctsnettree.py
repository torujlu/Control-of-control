import torch
import gc
import numpy as np
from torch.distributions import Categorical

class MCTSnetTree():
	
    def __init__(self, env, embedding_size, device = 'cpu'):

        self.__env = env
        self.__embedding_size = embedding_size
        self.__device = device
        self.__root = MCTSnetNode(None, self)
        env_state = env.get_state()
        state = torch.unsqueeze(torch.tensor(env_state[:2]), 0).to(device)
        reward = None #torch.zeros(1,1).to(device)
        self.__root.set_state(state)
        self.__root.set_reward(reward)
        self.__root.set_done(False)
        self.__root.set_probs_mask(env.get_mask())

    def get_root(self):
        return self.__root
        
    def set_root(self, root):
        del self.__root
        self.__root = root
        self.__root.set_parent(None)

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
	
    def __init__(self, parent, tree):

        self.__device = tree.get_device()
        #self.__action = action #torch.unsqueeze(torch.tensor([float(action)]), 0).to(self.__device) #, requires_grad=True
        self.__parent = parent
        self.__tree = tree
        self.__children = []
        self.__h = torch.zeros(1, tree.get_embedding_size(), requires_grad=True).to(self.__device) #
    
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

    """
    def set_action(self, action):
        self.__action = action
    
    def get_action(self):
        return self.__action
    """
    def get_done(self):
        return self.__done
    
    def set_done(self, done):
        self.__done = done

    def get_parent(self):
        return self.__parent
    
    def set_parent(self, parent):
        self.__parent = parent

    def expand(self):
        for action in range(len(self.__probs_mask)):
            if self.__probs_mask[action]:
                child = MCTSnetNode(self, self.__tree)
                self.__children.append((action, child))
    
    def get_children(self):
        return self.__children

    def get_probs_mask(self):
        return self.__probs_mask

    def set_probs_mask(self, probs_mask):
        self.__probs_mask = probs_mask
    
    def next_node(self, probs):
        self.set_probs(probs)
        probs = torch.squeeze(probs)
        n_grid, _, _ = self.__tree.get_env().get_dims()
        """
        block_mask = torch.tensor(np.any(self.__probs_mask, axis=(1,2)).astype(float)).to(self.__device)
        masked_block_probs = probs[:n_possible_blocks]*block_mask/torch.sum(probs[:n_possible_blocks]*block_mask)
        m1 = Categorical(masked_block_probs)
        block = m1.sample()
        y_mask = torch.tensor(np.any(self.__action_mask[block.item()], axis=1).astype(float)).to(self.__device)
        masked_y_probs = probs[n_possible_blocks:n_possible_blocks+n_grid]*y_mask/torch.sum(probs[n_possible_blocks:n_possible_blocks+n_grid]*y_mask)
        m2 = Categorical(masked_y_probs)
        y = m2.sample()
        x_mask = torch.tensor(self.__action_mask[block.item(),y.item()]).to(self.__device)
        masked_x_probs = probs[n_possible_blocks+n_grid:]*x_mask/torch.sum(probs[n_possible_blocks+n_grid:]*x_mask)
        m3 = Categorical(masked_x_probs)
        x = m3.sample()
        env_action = np.array([block.item(),y.item(),x.item()])
        action = env_action[0]*n_grid*n_grid + env_action[1]*n_grid + env_action[2]
        """

        probs_mask = torch.tensor(self.__probs_mask).to(self.__device)
        masked_probs = probs*probs_mask
        if ~np.any(masked_probs.clone().cpu().detach().numpy()):
            masked_probs = probs_mask     
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
                #child.set_action(action)
                child.set_done(done)
                child.set_probs_mask(self.__tree.get_env().get_mask())

                return child

    def delete(self):
        for (_, child) in self.__children:
            child.delete()
            del child
