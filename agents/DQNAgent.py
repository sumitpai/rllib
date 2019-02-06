from .QAgent import QAgent

import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os

class DQNAgent(QAgent):
    def __init__(self, state_features, action_size, model_class_dict={}, agent_params={}, rnd=-1):
        super().__init__("DQN", state_features, action_size, model_class_dict, agent_params, rnd)

    def _initialize_models(self):
        self.q_pred_net = self.model_class_dict["model_class"](self.state_features, 
                                                                      self.action_size, 
                                                                      self.model_class_dict["hidden_layers"]).to(self.device)
        self.q_target_net = self.model_class_dict["model_class"](self.state_features, 
                                                                        self.action_size, 
                                                                        self.model_class_dict["hidden_layers"]).to(self.device)
        self.optimizer = optim.Adam(self.q_pred_net.parameters(), 
                                    lr=self.agent_params.get('lr', 0.0001))
        
            
    def _learn(self, sample_experiences, gamma):
        self.q_target_net.eval()
        self.q_pred_net.train()
        
        states, actions, next_states, rewards, dones = self._convert_to_torch(sample_experiences)
        
        pred_Q = self.q_pred_net(states).gather(1, actions)
        
        with torch.no_grad():
            target_Q_next = self.q_target_net(next_states).detach()
            target_Q_next = target_Q_next.max(1)[0].unsqueeze(1)
            target_Q = rewards + (gamma * target_Q_next * (1-dones))
        
        
        loss = F.mse_loss(pred_Q, target_Q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_target_net()
        
    def save(self, save_dir='.'):
        torch.save(self.q_pred_net.state_dict(), os.path.join(save_dir, 'nav_agent_checkpoint.pth'))
        
    def load(self, load_dir='.'):
        chkpt = torch.load(os.path.join(load_dir, 'nav_agent_checkpoint.pth'))
        self.q_pred_net.load_state_dict(chkpt)
        
    def _update_target_net(self):
        for target_net_param, pred_net_param in zip(self.q_target_net.parameters(), self.q_pred_net.parameters()):
            target_net_param.data.copy_(self.tau * pred_net_param.data + 
                                        (1.0-self.tau) * target_net_param.data)
            
    def predict(self, state, epsilon=0.):
        self.q_pred_net.eval()
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_vals_pred = self.q_pred_net(state)
        
        if random.random() > epsilon:
            return np.argmax(q_vals_pred.cpu().data.numpy())
        else:
            return np.random.choice(self.action_size)
        