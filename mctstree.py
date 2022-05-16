import gc
import numpy as np


class MCTSTree():

    def __init__(self, env):

        self.__env = env
        self.__root = MCTSNode(None, self, -1)
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
    
    def delete(self):
        self.__root.delete()
        gc.collect()

class MCTSNode():

    def __init__(self, parent, tree, action):

        self.__action = action
        self.__parent = parent
        self.__tree = tree
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
                child = MCTSNode(self, self.__tree, action)
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

    def next_node(self, action):
        
        _, n_blocks, _, _, _ = self.__tree.get_env().get_dims()
        block = action//(3*n_blocks)
        loc = action - block*3*n_blocks
        env_action = np.array([block,loc])
        _, env_reward, done = self.__tree.get_env().step(env_action)

        for (child_id, child) in self.__children:
            if child_id == action:
                child.set_reward(env_reward)
                child.set_done(done)
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