from .QAgent import QAgent
from ..utils import OUNoise
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os

class ActorCriticAgent(QAgent):
    def __init__(self, state_features, action_size, model_class_dict={}, agent_params={}, rnd=-1):
        super().__init__("Actor_Critic", state_features, action_size, model_class_dict, agent_params, rnd)
        
    def _initialize_models(self):
        self.n_agents = self.agent_params['n_agents']
        
        self.q_actor_pred_net = self.model_class_dict["model_class"](self.state_features, 
                                                                      self.action_size, 
                                                                      self.model_class_dict["actor_hidden_layers"],
                                                                      out_activation='tanh').to(self.device)
        self.q_actor_target_net = self.model_class_dict["model_class"](self.state_features, 
                                                                        self.action_size, 
                                                                        self.model_class_dict["actor_hidden_layers"],
                                                                        out_activation='tanh').to(self.device)
        
        self.q_critic_target_net = self.model_class_dict["model_class"](self.state_features, 
                                                                      self.action_size, 
                                                                      self.model_class_dict["critic_hidden_layers"],
                                                                      critic_input_layer=self.model_class_dict["critic_input_layer"] ).to(self.device)
        self.q_critic_pred_net = self.model_class_dict["model_class"](self.state_features, 
                                                                        self.action_size, 
                                                                        self.model_class_dict["critic_hidden_layers"],
                                                                        critic_input_layer=self.model_class_dict["critic_input_layer"] ).to(self.device)
        
        

        
        self.optimizer_actor = optim.Adam(self.q_actor_pred_net.parameters(), 
                                    lr=self.agent_params.get('lr', 0.001))
        
        self.optimizer_critic = optim.Adam(self.q_critic_pred_net.parameters(), 
                                    lr=self.agent_params.get('lr', 0.001))
        
        self.noise = OUNoise((self.n_agents, self.action_size), self.seed)

        
    def step(self, state, action, next_state, reward, done):
        self.episode_counter +=1
        if self.n_agents == 1:
            self.replay_memory.add(state, action, next_state, reward, done)
        else:
            for i in range(self.n_agents):
                self.replay_memory.add(state[i,:], action[i,:], next_state[i,:], reward[i], done[i])

        if self.episode_counter % self.agent_params.get('update_interval', 4) \
            and self.replay_memory.get_size() >=  self.batch_size:
            experience_replay_sample = self.replay_memory.get_sample(self.batch_size)
            self._learn(experience_replay_sample, self.agent_params.get('gamma', 0.99))
            
    def _convert_to_torch(self, sample):
        states, actions, next_states, rewards, dones = sample
        states = torch.from_numpy(np.float32(np.vstack(states))).to(self.device)
        actions = torch.from_numpy(np.float32(np.vstack(actions))).to(self.device)
        next_states = torch.from_numpy(np.float32(np.vstack(next_states))).to(self.device)
        rewards = torch.from_numpy(np.float32(np.vstack(rewards))).to(self.device)
        dones = torch.from_numpy(np.float32(np.vstack(dones))).to(self.device)
        return (states, actions, next_states, rewards, dones)
        
            
    def _learn(self, sample_experiences, gamma):
        self.q_actor_target_net.eval()
        self.q_critic_target_net.eval()
        self.q_actor_pred_net.train()
        self.q_critic_pred_net.train()
        
        states, actions, next_states, rewards, dones = self._convert_to_torch(sample_experiences)
        
        with torch.no_grad():
            actions_for_next_states = self.q_actor_target_net(next_states).detach()
            target_Q_next = self.q_critic_target_net(next_states, actions_for_next_states).detach()
            target_Q = rewards + (gamma * target_Q_next * (1-dones))
            
        # Critic update
        pred_Q = self.q_critic_pred_net(states, actions)
        critic_loss = F.mse_loss(pred_Q, target_Q)
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        #actor update
        actions_pred = self.q_actor_pred_net(states)
        actor_loss = -self.q_critic_pred_net(states, actions_pred).mean()
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        self._update_target_net(self.q_actor_pred_net, self.q_actor_target_net)
        self._update_target_net(self.q_critic_pred_net, self.q_critic_target_net)
            
    def save(self, save_dir="."):
        torch.save(self.q_actor_pred_net.state_dict(), os.path.join(save_dir, 'actor_checkpoint.pth'))
        torch.save(self.q_critic_pred_net.state_dict(), os.path.join(save_dir, 'critic_checkpoint.pth'))
    
    def load(self, load_dir="."):
        actor_checkpoint = torch.load(os.path.join(load_dir, 'actor_checkpoint.pth'))
        critic_checkpoint = torch.load(os.path.join(load_dir, 'critic_checkpoint.pth'))
        self.q_actor_pred_net.load_state_dict(actor_checkpoint)
        self.q_critic_pred_net.load_state_dict(critic_checkpoint)
        
    def predict(self, state, add_noise=True, epsilon=0.):
        self.q_actor_pred_net.eval()
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            actions = self.q_actor_pred_net(state)
        
        actions = np.squeeze(actions.cpu().data.numpy())
            
        if add_noise:
            actions += np.squeeze(self.noise.sample())
            
        if np.random.rand() > epsilon:
            return np.clip(actions, -1, 1)   
        else:
            return np.float32(np.random.rand(self.n_agents, self.action_size) - 0.5) * 2.0
    
    
