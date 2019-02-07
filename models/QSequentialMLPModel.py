import torch
import torch.nn as nn

class QSequentialMLPModel(nn.Module):
    def __init__(self, state_features, action_size, hidden_layers=[], out_activation=None, critic_input_layer=-1):
        """
        if critic_input_layer == -1 then works as actor else it acts as critic
        """
        super(QSequentialMLPModel, self).__init__()
            
        self.model_layers = nn.ModuleList()  
        
        self.action_input_layer = critic_input_layer
        output_size = action_size
        if critic_input_layer>=0:
            output_size = 1
        
        if len(hidden_layers)<critic_input_layer:
            raise ValueError("Invalid Value for critic_input_layer")
        
        if self.action_input_layer==0:
            state_features += action_size
        
        if len(hidden_layers)==0:
            self.model_layers.append(nn.Linear(state_features, output_size))
        else:
            self.model_layers.append(nn.Linear(state_features, hidden_layers[0]))
            self.model_layers.append(nn.ReLU())
            for i in range(1, len(hidden_layers)):
                if i==self.action_input_layer:
                    self.model_layers.append(nn.Linear(hidden_layers[i-1] + action_size, hidden_layers[i]))
                else:
                    self.model_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.model_layers.append(nn.ReLU())
            self.model_layers.append(nn.Linear(hidden_layers[-1], output_size))
            
        if out_activation==None:
            pass
        elif out_activation=='tanh':
            self.model_layers.append(nn.Tanh())
        elif out_activation=='sigmoid':
            self.model_layers.append(nn.Sigmoid())
        elif out_activation=='softmax':
            self.model_layers.append(nn.Softmax())
    
    def forward(self, *input_vars):
        concat = False
        state_features = input_vars[0]
        for i in range(len(self.model_layers)):
            if i == (2 * self.action_input_layer) and not concat: # to account for relu at each layer
                state_features = torch.cat((state_features, input_vars[1]), dim=1)
                concat = True
            state_features = self.model_layers[i](state_features)
        #print('done')
        return state_features