from controllers.deepqcontroller import Deepqcontroller
from controllers.neuralqcontroller import Neuralqcontroller
from controllers.qcontroller import Qcontroller
from controllers.controller import Controller
from env import Env
import gym
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # env = Env('CartPole-v1')
    env = gym.make('MountainCar-v0')  #specify environment to use
    control = Deepqcontroller(env)  #specify controller to use
    
    #Pick rendering preferences
    train_render = False    
    test_render = False

    #Pick number of runs for train and test
    n_episodes = 250 
    nr_tests = 5
    test_intervals = 10

    test_rewards = []

    for e in range(n_episodes):
        obs = env.reset()
        done = False

        while done==False:
            if train_render == True:
                env.render()
            action = control.generateAction(obs, e)
            obs, reward ,done,_ = env.step(action)
            control.update_q_values(e, obs, reward, done)
            # control.updateQtable(e, reward, obs)
        
        if  e % test_intervals == 0:
            test_reward = 0
            for i in range(nr_tests):
                obs = env.reset()
                done = False
                while done==False:
                    if test_render == True:
                        env.render()
                    action = control.generateAction(obs, e)
                    obs, reward ,done,_ = env.step(action)
                    test_reward += reward
            test_rewards.append(test_reward/nr_tests)

    print("----Finished Run----")
    print(f"test rewards: {test_rewards}")
    plt.plot(test_rewards)
    plt.show()

