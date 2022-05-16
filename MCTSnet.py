#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.optim as optim
import os
from torch.distributions import Categorical
from pathlib import Path
import glob
import gc

from mctsnet import MCTSnet
from mnetwork import MNetwork
from internal_m import InternalM
from tangram import Tangram
from mctsnettree import MCTSnetTree
from itertools import count
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

gamma=0.9
alpha=0.05
tau=0.1
M_penalty=-10
seed=543
render=False
log_interval=1
gpu=True
load_agent=True
meta_control=False
internal_control=False
load_M=True

n_grid = 20
n_blocks = 4
n_possible_blocks = 6
chunk_type = 7
n_blocks_H = 0

embedding_size = 128
readout_hidden_size = 128
backup_hidden_size = 128
action_dims = [3,n_blocks,n_possible_blocks]
state_dims = [2,n_grid,n_grid]
embedding_n_residual_blocks = 3
embedding_channel_sizes = [64,64,64,32]
embedding_kernels = [3,3,3,1]
embedding_strides = [1,1,1,1]
policy_n_residual_blocks = 2
policy_channel_sizes = [32,32,32,16]
policy_kernels = [3,3,3,1]
policy_strides = [1,1,1,1]
policy_hidden_size = 128
M_n_residual_blocks = 2
M_channel_sizes = [2,2,2,1]
M_kernels = [3,3,3,1]
M_strides = [1,1,1,1]
M_hidden_size = 8
internal_m_hidden_size = 8
max_M = 20
M = None

n_simuls = 10
n_evals = 100

serialization_path = './models/mctsnet/hierarchical_blocks_{}'.format(n_blocks_H)
print('serialization_path: ',serialization_path)
serialize_every_n_episodes = 10000
update_every_n_episodes = 1
test_every_n_episodes = 100
# create folder 
Path(serialization_path).mkdir(parents=True, exist_ok=True)


# In[3]:


device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False
torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)
filesave_paths_mctsnet = sorted(glob.glob(f'{serialization_path}/mctsnet_e*'))
filesave_paths_M = sorted(glob.glob(f'{serialization_path}/M_e*'))
if load_agent and len(filesave_paths_mctsnet) > 0:
    mctsnet = torch.load(open(filesave_paths_mctsnet[-1],'rb'), map_location=device)
    n_episodes = int(filesave_paths_mctsnet[-1][48:54])
    running_reward = float(filesave_paths_mctsnet[-1][56:].replace('.pt',''))
    print('Loaded MCTSnet from '+ filesave_paths_mctsnet[-1])
else:
    mctsnet = MCTSnet(embedding_size,
                      readout_hidden_size,
                      backup_hidden_size,
                      action_dims,
                      state_dims,
                      embedding_n_residual_blocks,
                      embedding_channel_sizes,
                      embedding_kernels,
                      embedding_strides,
                      policy_n_residual_blocks,
                      policy_channel_sizes,
                      policy_kernels,
                      policy_strides,
                      policy_hidden_size,
                      device).to(device)
    n_episodes = 0
    running_reward = -1
    print('Initialized new MCTSnet')
if meta_control:
    if load_M and len(filesave_paths_M) > 0:
        M = torch.load(open(filesave_paths_M[-1],'rb'), map_location=device)
        print('Loaded MNetwork from '+ filesave_paths_M[-1])
    elif internal_control:
        M = InternalM(embedding_size, internal_m_hidden_size).to(device)
    else:
        M = MNetwork(state_dims,
                     M_n_residual_blocks,
                     M_channel_sizes,
                     M_kernels,
                     M_strides,
                     M_hidden_size,
                     max_M).to(device)
        print('Initialized new MNetwork')
seed += n_episodes
env = Tangram(seed, n_grid, n_blocks, n_possible_blocks, chunk_type, n_blocks_H)
tree = MCTSnetTree(env, embedding_size, device)
#optimizer_mctsnet = optim.Adam(mctsnet.parameters(), lr=5e-4)
optimizer_mctsnet = optim.SGD(mctsnet.parameters(), lr=5e-4)
if meta_control:
    #optimizer_M = optim.Adam(M.parameters(), lr=5e-4)
    optimizer_M = optim.SGD(M.parameters(), lr=5e-4)


# In[4]:


def select_action(tree, n_simuls, mctsnet):
    probs, action_mask = mctsnet(tree, M, n_simuls, gamma, internal_control)    
    probs = torch.squeeze(probs)
    action_mask = torch.tensor(action_mask).to(device)
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
    mctsnet.update_saved_log_probs(m.log_prob(action))
    mctsnet.update_saved_entropies(m.entropy())
    return env_action, action_id, m.log_prob(action)


# In[5]:


def finish_episode(mctsnet, optimizer, episode_num, *args):
    R = 0
    mctsnet_loss = []
    returns = []
    for r in mctsnet.get_rewards()[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns).to(device)
    for log_prob, R, entropy in zip(mctsnet.get_saved_log_probs(), returns, mctsnet.get_saved_entropies()):
        mctsnet_loss.append(torch.unsqueeze(-log_prob*R - tau*entropy,0))
    mctsnet_loss = torch.cat(mctsnet_loss).sum()/update_every_n_episodes
    if episode_num % update_every_n_episodes == 0:
        optimizer.zero_grad()
    mctsnet_loss.backward()
    #torch.nn.utils.clip_grad_norm_(mctsnet.parameters(), 1.0)
    if episode_num % update_every_n_episodes == 0:
        optimizer.step()
        #print('Updated Weights!')
    mctsnet.update_losses(mctsnet_loss.clone().cpu().detach().numpy())
    mctsnet.delete_rewards()
    mctsnet.delete_saved_log_probs()
    mctsnet.delete_saved_entropies()
    if meta_control:
        M = args[0]
        optimizer_M = args[1]
        R = 0
        M_loss = []
        returns = []
        for r in M.get_rewards()[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)
        for log_prob, R in zip(M.get_saved_log_probs(), returns):
            M_loss.append(torch.unsqueeze(-log_prob*R,0))
        optimizer_M.zero_grad()
        M_loss = torch.cat(M_loss).sum()
        M_loss.backward()
        #torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0)
        optimizer_M.step()
        M.update_losses(M_loss.clone().cpu().detach().numpy())
        M.delete_rewards()
        M.delete_saved_log_probs()


# In[ ]:


reward_threshold = 0.99
success_threshold = 99
success_ratio = 0
for i_episode in count(n_episodes+1):
    env.reset()
    if render:
        env.render()
    tree = MCTSnetTree(env, embedding_size, device)
    ep_reward = 0
    for t in range(1, n_blocks + 1):
        if meta_control and not internal_control:
            M_probs = M(tree.get_root().get_state())
            M_probs = torch.squeeze(M_probs)
            M_m = Categorical(M_probs)
            M_action = M_m.sample()
            n_simuls = M_action.item()+1
            M.update_saved_log_probs(M_m.log_prob(M_action))
        env_action, action, log_prob = select_action(tree, n_simuls, mctsnet) #probs
        if meta_control and internal_control:
            n_simuls = M.get_saved_n_simuls()[-1]
        env_state, env_reward, done = env.step(env_action)
        state = torch.unsqueeze(torch.tensor(env_state[:2]), 0).to(device)
        reward = torch.unsqueeze(torch.tensor([env_reward]), 0).to(device)
        for (child_id, child) in tree.get_root().get_children():
            if child_id == action:
                child.set_state(state)
                child.set_reward(reward)
                child.set_done(done)
                child.set_action(torch.reshape(-log_prob, (1,1))) #torch.reshape(probs[action], (1,1))
                child.set_action_mask(tree.get_env().get_mask())
                tree.set_root(child)
                break
        if render:
            env.render()
        mctsnet.update_rewards(reward)
        if meta_control:
            if env_reward > 0:
                M_reward = 1-alpha*n_simuls
            elif env_reward < 0:
                M_reward = M_penalty
            else:
                M_reward = 0
            M_reward = torch.unsqueeze(torch.tensor([M_reward]), 0).to(device)
            M.update_rewards(M_reward)
        ep_reward += env_reward
        if done:
            break

    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    mctsnet.update_running_rewards(running_reward)
    if meta_control:
        finish_episode(mctsnet, optimizer_mctsnet, i_episode, M, optimizer_M)
    else:
        finish_episode(mctsnet, optimizer_mctsnet, i_episode)

    if i_episode % log_interval == 0:
        if meta_control:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tLast loss: {:.2f}   \tLast M: {}\tLast M loss: {:.2f}'.format(
                    i_episode, ep_reward, running_reward, mctsnet.get_last_episode_loss(), n_simuls, M.get_last_episode_loss()))
        else:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tLast loss: {:.2f}'.format(
                i_episode, ep_reward, running_reward, mctsnet.get_last_episode_loss()))
    if serialize_every_n_episodes > 0 and i_episode % serialize_every_n_episodes == 0:
        torch.save(mctsnet, f"{serialization_path}/mctsnet_e{str(i_episode).zfill(6)}_p{running_reward}.pt")
        print("Saved the model!")
        del mctsnet, optimizer_mctsnet
        filesave_paths_mctsnet = sorted(glob.glob(f'{serialization_path}/mctsnet_e*'))
        mctsnet = torch.load(open(filesave_paths_mctsnet[-1],'rb'), map_location=device)
        n_episodes = int(filesave_paths_mctsnet[-1][48:54])
        running_reward = float(filesave_paths_mctsnet[-1][56:].replace('.pt',''))
        print('Loaded MCTSnet from '+ filesave_paths_mctsnet[-1])
        #optimizer_mctsnet = optim.Adam(mctsnet.parameters(), lr=5e-4)
        optimizer_mctsnet = optim.SGD(mctsnet.parameters(), lr=5e-4)
        if meta_control:
            torch.save(M, f"{serialization_path}/M_e{str(i_episode).zfill(6)}_p{running_reward}.pt")
            del M, optimizer_M
            filesave_paths_M = sorted(glob.glob(f'{serialization_path}/M_e*'))
            M = torch.load(open(filesave_paths_M[-1],'rb'), map_location=device)
            print('Loaded M from '+ filesave_paths_M[-1])
            #optimizer_M = optim.Adam(M.parameters(), lr=5e-4)
            optimizer_M = optim.SGD(M.parameters(), lr=5e-4)
        gc.collect()
        torch.cuda.empty_cache()
    if i_episode % test_every_n_episodes == 0:
        print('Testing...')
        mctsnet.eval()
        if meta_control:
            M.eval()
        with torch.no_grad():
            success_ratio = 0
            for eval_num in range(1,n_evals+1):
                env.reset()
                tree = MCTSnetTree(env, embedding_size, device)
                done = False
                while not done:
                    if render:
                        env.render()
                    if meta_control and not internal_control:
                        M_probs = M(tree.get_root().get_state())
                        M_probs = torch.squeeze(M_probs)
                        #M_m = Categorical(M_probs)
                        #M_action = M_m.sample()
                        #n_simuls = M_action.item()+1
                        n_simuls = torch.argmax(M_probs).item()+1
                    probs, action_mask = mctsnet(tree, M, n_simuls, gamma, internal_control)
                    probs = torch.squeeze(probs)
                    action_mask = torch.tensor(action_mask).to(device)
                    masked_probs = probs*action_mask 
                    if not torch.sum(masked_probs.clone()) > 0:
                        masked_probs += action_mask
                    masked_probs /= torch.sum(masked_probs)
                    m = Categorical(masked_probs)
                    #action = m.sample().item()
                    action = torch.argmax(masked_probs)
                    action_id = action.item()
                    block = action_id//(3*n_blocks)
                    loc = action_id - block*3*n_blocks
                    env_action = np.array([block,loc])
                    if meta_control and internal_control:
                        n_simuls = M.get_saved_n_simuls()[-1]
                    env_state, env_reward, done = env.step(env_action)

                    state = torch.unsqueeze(torch.tensor(env_state[:2]), 0).to(device)
                    reward = torch.unsqueeze(torch.tensor([env_reward]), 0).to(device)
                    for (child_id, child) in tree.get_root().get_children():
                        if child_id == action_id:
                            child.set_state(state)
                            child.set_reward(reward)
                            child.set_done(done)
                            child.set_action(torch.reshape(-m.log_prob(action), (1,1))) #torch.reshape(probs[action_id], (1,1))
                            child.set_action_mask(tree.get_env().get_mask())
                            tree.set_root(child)
                            break
                if env_reward == 1:
                    success_ratio += 1
                if render:
                    env.render()
            success_ratio *= 100/n_evals
            print("Success ratio: {}%".format(success_ratio))
            mctsnet.update_success_ratios(success_ratio)
        mctsnet.train()
        if meta_control:
            M.train()
    if running_reward > reward_threshold or success_ratio > success_threshold:
        print("Solved! Running reward is now {} and "
                "the last episode runs to {} time steps!".format(running_reward, t))
        torch.save(mctsnet, f"{serialization_path}/mctsnet_e{str(i_episode).zfill(6)}_p{running_reward}.pt")
        if meta_control:
            torch.save(M, f"{serialization_path}/M_e{str(i_episode).zfill(6)}_p{running_reward}.pt")
        print("Saved the model!")
        break
    tree.delete()
    del tree
    gc.collect()
    torch.cuda.empty_cache()


# In[ ]:


torch.save(mctsnet, f"{serialization_path}/mctsnet_e{str(i_episode).zfill(6)}_p{running_reward}.pt")
if meta_control:
    torch.save(M, f"{serialization_path}/M_e{str(i_episode).zfill(6)}_p{running_reward}.pt")

