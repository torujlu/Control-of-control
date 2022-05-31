import torch
import gc
import numpy as np
from torch.distributions import Categorical

class MCTS2Tree():

    def __init__(self, env, device = 'cpu'):

        self.__env = env
        self.__device = device
        self.__root = MCTS2Node(None, self, -1)
        self.__root.set_reward(0)
        self.__root.set_done(False)
        self.__root.set_action_mask(env.get_mask())
    
    def get_root(self):
        return self.__root
    
    def set_root(self, root):
        del self.__root
        self.__root = root
        self.__root.set_parent(None)

    def get_env(self):
        return self.__env
    
    def get_device(self):
        return self.__device
    
    def delete(self):
        self.__root.delete()
        torch.cuda.empty_cache()
        gc.collect()

class MCTS2Node():

    def __init__(self, parent, tree, action):
        
        self.__device = tree.get_device()
        self.__parent = parent
        self.__tree = tree
        self.__action = action
        _, n_blocks, n_possible_blocks, _, _ = tree.get_env().get_dims()
        self.__children = []
        self.__Q = np.zeros(3*n_blocks*n_possible_blocks)
        self.__N = np.zeros(3*n_blocks*n_possible_blocks)
    
    def get_reward(self):
        return self.__reward

    def set_reward(self, reward):
        self.__reward = reward

    def get_action(self):
        return self.__action

    def get_done(self):
        return self.__done

    def set_done(self, done):
        self.__done = done

    def get_parent(self):
        return self.__parent

    def set_parent(self, parent):
        self.__parent = parent
    
    def expand(self):
        for action in range(len(self.__action_mask)):
            if self.__action_mask[action]:
                child = MCTS2Node(self, self.__tree, action)
                self.__children.append((action, child))
    
    def get_children(self):
        return self.__children

    def get_action_mask(self):
        return self.__action_mask

    def set_action_mask(self, action_mask):
        self.__action_mask = action_mask

    def get_Q(self):
        return self.__Q
    
    def set_Q(self, Q):
        self.__Q = Q
    
    def get_N(self):
        return self.__N
    
    def set_N(self, N):
        self.__N = N
    
    def get_log_prob(self):
        return self.__log_prob

    def set_log_prob(self, log_prob):
        self.__log_prob = log_prob
    
    def get_entropy(self):
        return self.__entropy
    
    def set_entropy(self, entropy):
        self.__entropy = entropy

    def next_node(self, probs):
        probs = torch.squeeze(probs).to(self.__device)
        _, n_blocks, _, _, _ = self.__tree.get_env().get_dims()
        action_mask = torch.tensor(self.__action_mask).to(self.__device)
        masked_probs = probs*action_mask
        if not torch.sum(masked_probs.clone()) > 0:
            masked_probs += action_mask   
        masked_probs /= torch.sum(masked_probs)
        m = Categorical(masked_probs)
        action = m.sample()
        action_id = action.item()
        block = action_id//(3*n_blocks)
        loc = action_id - block*3*n_blocks
        env_action = np.array([block,loc])
        #train_probs = masked_probs + 1e-4*(1-action_mask)
        #train_probs /= torch.sum(train_probs)
        #m_train = Categorical(train_probs)

        _, env_reward, done = self.__tree.get_env().step(env_action)
        for (child_id, child) in self.__children:
            if child_id == action:
                child.set_reward(env_reward)
                child.set_done(done)
                child.set_log_prob(m.log_prob(action)) #probs[action_id]
                child.set_entropy(m.entropy())
                child.set_action_mask(self.__tree.get_env().get_mask())

                return child
    
    def delete(self):
        for (_, child) in self.__children:
            child.delete()
            del child

    def roll_out(self, gamma):
        done = self.get_done()
        time = 0
        reward = 0
        while not done:
            action_mask = self.__tree.get_env().get_mask()
            action = np.random.choice(np.nonzero(action_mask)[0])
            _, n_blocks, _, _, _ = self.__tree.get_env().get_dims()
            block = action//(3*n_blocks)
            loc = action - block*3*n_blocks
            env_action = np.array([block,loc])

            _, env_reward, done = self.__tree.get_env().step(env_action)
            time += 1
            reward += (gamma**time)*env_reward
            
        self.__tree.get_env().revert(time)
        
        return self.get_reward() + reward