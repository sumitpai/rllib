import torch
import torch.nn as nn

class QSequentialMLPModel(nn.Module):
    def __init__(self, state_features, action_size, hidden_layers=[]):
        super(QSequentialMLPModel, self).__init__()
            
        self.model_layers = nn.ModuleList()    
        if len(hidden_layers)==0:
            self.model_layers.append(nn.Linear(state_features, action_size))
        else:
            self.model_layers.append(nn.Linear(state_features, hidden_layers[0]))
            self.model_layers.append(nn.ReLU())
            for i in range(1, len(hidden_layers)):
                self.model_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.model_layers.append(nn.ReLU())
            self.model_layers.append(nn.Linear(hidden_layers[-1], action_size))
    
    def forward(self, state_features):
        for i in range(len(self.model_layers)):
            #print(state_features.size())
            state_features = self.model_layers[i](state_features)
        #print('done')
        return state_features