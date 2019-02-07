import abc
import numpy as np
import random
import torch
import os
import torch.optim as optim

from ..utils import ReplayMemory

class QAgent(abc.ABC):
    def __init__(self, agent_name, state_features, action_size, model_class_dict, agent_params, rnd=-1):
        self.seed = rnd
        if rnd!=-1:
            np.random.seed(rnd)
            random.seed(rnd)
            torch.manual_seed(rnd)
            
        self.episode_counter = 0
        self.state_features = state_features
        self.action_size = action_size
        self.agent_params = agent_params
        self.model_class_dict = model_class_dict
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print('Using:', self.device)
        
        self.batch_size = self.agent_params.get('min_batch_size', 64)
        self.tau = self.agent_params.get('tau', 0.001)
        
        self._initialize_models()
        
        self.replay_memory = ReplayMemory(action_size,
                                  agent_params.get('memory_size', 50000),
                                      rnd)
        
    def _initialize_models(self):
        raise NotImplementedError('Not implemented')
        
    def _convert_to_torch(self, sample):
        states, actions, next_states, rewards, dones = sample
        states = torch.from_numpy(np.float32(np.vstack(states))).to(self.device)
        actions = torch.from_numpy(np.int64(np.vstack(actions))).to(self.device)
        next_states = torch.from_numpy(np.float32(np.vstack(next_states))).to(self.device)
        rewards = torch.from_numpy(np.float32(np.vstack(rewards))).to(self.device)
        dones = torch.from_numpy(np.float32(np.vstack(dones))).to(self.device)
        return (states, actions, next_states, rewards, dones)
        
    def step(self, state, action, next_state, reward, done):
        self.episode_counter +=1
        self.replay_memory.add(state, action, next_state, reward, done)
        if self.episode_counter % self.agent_params.get('update_interval', 4) \
            and self.replay_memory.get_size() >=  self.batch_size:
            experience_replay_sample = self.replay_memory.get_sample(self.batch_size)
            self._learn(experience_replay_sample, self.agent_params.get('gamma', 0.99))
            
    def _learn(self, sample_experiences, gamma):
        raise NotImplementedError('Not implemented')
        
    def save(self, save_dir='.'):
        raise NotImplementedError('Not implemented')
        
    def load(self, load_dir='.'):
        raise NotImplementedError('Not implemented')
        
    def _update_target_net(self, q_pred_net, q_target_net):
        for target_net_param, pred_net_param in zip(q_target_net.parameters(), q_pred_net.parameters()):
            target_net_param.data.copy_(self.tau * pred_net_param.data + 
                                        (1.0-self.tau) * target_net_param.data)
        
    def predict(self, state, exploration_params):
        raise NotImplementedError('Not implemented')