import controllers.controller 
import math
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense, Activation, Flatten
# import torchvision.transforms as transforms
# import torchvision.datasets as dsets

class Neuralqcontroller(controllers.controller.Controller):
    
    """
    Neural q learning controller
    """
    def __init__(self, env):
        self.env = env
        self.model = self.create_model()
    
    def create_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.env.env.observation_space.shape))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.env.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        return model

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
            action = np.argmax(self.model.predict(np.identity(self.env.env.observation_space.shape)[obs:obs + 1])) # exploit
        
        return self.action
    
    def updateQtable(self, e, reward, obs):
        new_state = self.discretizer(*obs)
        lr = self.learning_rate(e)
        learnt_value = self.new_Q_value(reward , new_state )
        old_value = self.Q_table[self.current_state][self.action]
        self.Q_table[self.current_state][self.action] = (1-lr)*old_value + lr*learnt_value     