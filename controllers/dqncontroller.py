import controllers.controller 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from collections import deque 
import numpy as np
import math

class Dqncontroller(controllers.controller.Controller):

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.state_shape = env.observation_space.shape
        self.action_shape = env.action_space.n
        
        self.model = self.create_model()

        self.pretrain_length = 64

        self.memory_size = 1000000          # Number of experiences the Memory can keep
        self.memory = Memory(max_size = self.memory_size)
        self.prepopulate_memory()

    
    def generateAction(self, obs, e):
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
        # ## EPSILON GREEDY STRATEGY
        # # Choose action a from state s using epsilon greedy.
        # ## First we randomize a number
        # exp_exp_tradeoff = np.random.rand()

        # # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        # explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        
        # if (explore_probability > exp_exp_tradeoff):
        #     # Make a random action (exploration)
        #     choice = random.randint(1,len(possible_actions))-1
        #     action = possible_actions[choice]
            
        # else:
        #     # Get action from Q-network (exploitation)
        #     # Estimate the Qs values state
        #     Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
            
        #     # Take the biggest Q value (= the best action)
        #     choice = np.argmax(Qs)
        #     action = possible_actions[choice]
                
                
        # return action, explore_probability

        

    def exploration_rate(self, n : int, min_rate= 0.01 ) -> float :
        """Decaying exploration rate"""
        return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))

    def create_model(self):
        learning_rate = 0.001
        init = tf.keras.initializers.HeUniform()
        model = Sequential()
        model.add(Dense(24, input_shape=self.state_shape, activation='relu', kernel_initializer=init))
        model.add(Dense(12, activation='relu', kernel_initializer=init))
        model.add(Dense(self.action_shape, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
        return model
    
    def prepopulate_memory(self):
        # Instantiate memory
        memory = Memory(max_size = self.memory_size)
        for i in range(self.pretrain_length):
            # If it's the first step
            if i == 0:
                state = self.env.reset()
                                
            # Get the next_state, the rewards, done by taking a random action
            self.action = self.env.action_space.sample() # explore 
            next_state, reward, done, _ = self.env.step(self.action)
            
            # If the episode is finished (we're dead 3x)
            if done:
                # We finished the episode
                next_state = np.zeros(state.shape)
                
                # Add experience to memory
                memory.add((state, self.action, reward, next_state, done))
                
                # Start a new episode
                state = self.env.reset()
                
            else:
                # Add experience to memory
                memory.add((state, self.action, reward, next_state, done))
                
                # Our new state is now the next_state
                state = next_state

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]