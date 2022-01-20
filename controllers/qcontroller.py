import controllers.controller 
import math
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple
import numpy as np

class Qcontroller(controllers.controller.Controller):
    
    """
    Basic q learning controller
    """
    def __init__(self, env):
        super().__init__()
        self.n_bins = ( 6 , 12 )
        self.lower_bounds = [ env.observation_space.low[2], -math.radians(50) ]
        self.upper_bounds = [ env.observation_space.high[2], math.radians(50) ]
        self.Q_table = np.zeros(self.n_bins + (env.action_space.n,))
        self.env = env
        self.name = "regq"

    def discretizer(self, _ , __ , angle, pole_velocity ) -> Tuple[int,...]:
        """Convert continues state intro a discrete state"""
        est = KBinsDiscretizer(n_bins= self.n_bins, encode='ordinal', strategy='uniform')
        est.fit([self.lower_bounds, self.upper_bounds ])
        
        return tuple(map(int,est.transform([[angle, pole_velocity]])[0]))

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
        self.current_state = self.discretizer(*obs)

        # policy action 
        self.action = self.policy(self.current_state) # exploit
        
        # insert random action
        if np.random.random() < self.exploration_rate(e) : 
            self.action = self.env.action_space.sample() # explore 
        
        return self.action
    
    def updateQtable(self, e, reward, obs):
        new_state = self.discretizer(*obs)
        lr = self.learning_rate(e)
        learnt_value = self.new_Q_value(reward , new_state )
        old_value = self.Q_table[self.current_state][self.action]
        self.Q_table[self.current_state][self.action] = (1-lr)*old_value + lr*learnt_value     