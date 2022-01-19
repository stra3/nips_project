import controllers.controller 
import math
import random
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense, Activation, Flatten
from collections import deque

class Deepqcontroller(controllers.controller.Controller):
    
    """
    deep q learning controller
    """
    def __init__(self, env):
        self.env = env
        self.state_shape = env.observation_space.shape
        self.action_shape = env.action_space.n
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=50_000)
        self.target_update_counter = 0
        self.steps_to_update_target_model = 0
        self.total_training_rewards = 0

        self.action = None
        self.obs = None

        self.name="deepq"


    def create_model(self):
            """ The agent maps X-states to Y-actions
            e.g. The neural network output is [.1, .7, .1, .3]
            The highest value 0.7 is the Q-Value.
            The index of the highest action (0.7) is action #1.
            """
            learning_rate = 0.001
            init = tf.keras.initializers.HeUniform()
            model = Sequential()
            model.add(Dense(24, input_shape=self.state_shape, activation='relu', kernel_initializer=init))
            model.add(Dense(12, activation='relu', kernel_initializer=init))
            model.add(Dense(self.action_shape, activation='linear', kernel_initializer=init))
            model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
            return model

    def generateAction(self, obs, e):
        self.steps_to_update_target_model += 1
        # insert random action
        if np.random.random() < self.exploration_rate(e) : 
            self.action = self.env.action_space.sample() # explore 
        # policy action
        else:
            encoded = obs
            encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
            predicted = self.model.predict(encoded_reshaped).flatten()
            self.action = np.argmax(predicted) # exploit
        self.obs = obs
        
        return self.action

    def update_q_values(self, e, new_observation, reward, done):
        self.replay_memory.append([self.obs, self.action, reward, new_observation, done])
        # 3. Update the Main Network using the Bellman Equation
        if self.steps_to_update_target_model % 4 == 0 or done:
            self.train(done)

        self.observation = new_observation
        self.total_training_rewards += reward

        if done:
            if e % 10 == 0:
                print('Total training rewards: {} epoch = {}'.format(self.total_training_rewards, e))
            self.total_training_rewards += 1

            if self.steps_to_update_target_model >= 100:
                print('Copying main network weights to the target network weights')
                self.target_model.set_weights(self.model.get_weights())
                self.steps_to_update_target_model = 0
            self.total_training_rewards = 0
            

    
    def train(self, done):
        learning_rate = 0.7 # Learning rate
        discount_factor = 0.6

        MIN_REPLAY_SIZE = 1000
        if len(self.replay_memory) < MIN_REPLAY_SIZE:
            return

        batch_size = 64 * 2
        mini_batch = random.sample(self.replay_memory, batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []
        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + discount_factor * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

            X.append(observation)
            Y.append(current_qs)
        self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)
        


    def new_Q_value(self, reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
        """Temperal diffrence for updating Q-value of state-action pair"""
        future_optimal_value = np.max(self.Q_table[new_state])
        learned_value = reward + discount_factor * future_optimal_value
        return learned_value

    # Adaptive learning of Learning Rate
    def learning_rate(self, n : int , min_rate=0.01 ) -> float  :
        """Decaying learning rate"""
        return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))
    
    def exploration_rate(self, n : int, min_rate= 0.01 ) -> float :
        """Decaying exploration rate"""
        return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))
