import controllers.controller 
import math
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
# import torchvision.transforms as transforms
# import torchvision.datasets as dsets

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.tanh = nn.Tanh()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function
        out = self.fc1(x)
        # Non-linearity
        out = self.tanh(out)
        # Linear function (readout)
        out = self.fc2(out)
        output = torch.sigmoid(out,)

        return output

class Neuralqcontroller(controllers.controller.Controller):
    
    """
    Neural q learning controller
    """
    def __init__(self, env):
        self.env = env
        input_dim = 4
        hidden_dim = 100
        output_dim = 1
        self.model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
   
        self.criterion = nn.CrossEntropyLoss()

        learning_rate = 0.1

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        

    def policy(self, state : tuple ):
        """Choosing action based on epsilon-greedy policy"""
        return np.argmax(self.Q_table[state])

    def new_Q_value(self, reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
        """Temperal diffrence for updating Q-value of state-action pair"""
        future_optimal_value = np.max(self.Q_table[new_state])
        learned_value = reward + discount_factor * future_optimal_value
        return learned_value

    # Adaptive learning of Learning Rate
    def learning_rate(self, n : int , min_rate=0.01 ) -> float  :
        """Decaying learning rate"""
        return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))
    
    def exploration_rate(self, n : int, min_rate= 0.1 ) -> float :
        """Decaying exploration rate"""
        return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))

    def generateAction(self, obs, e):
        
        # insert random action
        if np.random.random() < self.exploration_rate(e) : 
            self.action = np.random.randint(2) # explore 
        # policy action
        else:
            prediction = self.model(torch.Tensor(obs))
            self.action = round(prediction.item()) # exploit
        
        return self.action
    
    def updateQtable(self, e, reward, obs):
        new_state = self.discretizer(*obs)
        lr = self.learning_rate(e)
        learnt_value = self.new_Q_value(reward , new_state )
        old_value = self.Q_table[self.current_state][self.action]
        self.Q_table[self.current_state][self.action] = (1-lr)*old_value + lr*learnt_value     