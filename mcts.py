import numpy as np


class MCTS():

    def __init__(self):

        pass

    def run_simulation(self, tree, n_simuls = 10, c = 1.0, gamma = 0.9):
        
        for _ in range(n_simuls):
            
            current_node = tree.get_root()
            done = current_node.get_done()
            children = current_node.get_children()

            while children and not done:

                Q = current_node.get_Q()
                N = current_node.get_N()
                action_mask = current_node.get_action_mask()
                masked_Q = action_mask*Q + (action_mask - 1)*1e6
                """
                masked_N = action_mask*N + (action_mask - 1)
                unexplored = np.where(masked_N==0)[0]
                if len(unexplored)>0:
                    action = np.random.choice(unexplored)
                else:
                """
                for action in range(len(N)):
                    #if N[action] > 0:
                    masked_Q[action] += c*np.sqrt(np.log(np.sum(N)+1)/(N[action]+1))
                    """
                    elif np.sum(N) > 0:
                        masked_Q[action] += c*np.sqrt(np.log(np.sum(N)))
                    """
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
        #N = current_node.get_N()
        Q = current_node.get_Q()
        action_mask = current_node.get_action_mask()
        #masked_N = action_mask*N + (action_mask - 1)
        masked_Q = action_mask*Q + (action_mask - 1)*1e6
        action = np.random.choice(np.where(masked_Q==np.max(masked_Q))[0])

        return action