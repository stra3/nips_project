from controllers.deepqcontroller import Deepqcontroller
from controllers.dqncontroller import Dqncontroller
from controllers.neuralqcontroller import Neuralqcontroller
from controllers.qcontroller import Qcontroller
from controllers.controller import Controller
from controllers.randomcontroller import Randomcontroller
from env import Env
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import time
import os

if __name__ == '__main__':
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    run_name = f"base-regq{int(time.time())}"

    #specify environment to use
    # env = gym.make('LunarLander-v2')  
    env = gym.make('CartPole-v1')

    #specify controller to use
    control = Randomcontroller(env)
    # control = Qcontroller(env)
    # control = Deepqcontroller(env)  
    
    #Pick rendering preferences
    train_render = False    
    test_render = False

    #Pick number of runs for train and test
    n_episodes = 500 
    nr_tests = 5
    test_intervals = 10

    run_name = f"{control.name}-env{env.unwrapped.spec.id}-eps{n_episodes}-tests{nr_tests}-testinterval{test_intervals}-time{int(time.time())}"

    test_rewards = []

    for e in range(n_episodes):
        obs = env.reset()
        done = False

        while done==False:
            if train_render == True:
                env.render()
            action = control.generateAction(obs, e)
            obs, reward ,done,_ = env.step(action)
            
            if control.name == "regq":
                control.updateQtable(e, reward, obs) #function for updating Q in Qlearning
            else:
                control.update_q_values(e, obs, reward, done) #function for updating Q in deepQ

        if  e % test_intervals == 0:
            print(f"Test: {e/test_intervals}/{np.floor(n_episodes/test_intervals)}")
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
            print(f"Test reward average: {test_reward/nr_tests}")
            test_rewards.append(test_reward/nr_tests)

    print("----Finished Run----")
    print(f"test rewards: {test_rewards}")
    plt.plot(test_rewards)
    plt.show()

    # open the file in the write mode
    with open(f'output/{run_name}', 'w') as f:
    # create the csv writer
        writer = csv.writer(f)

    # write a row to the csv file
        writer.writerow(test_rewards)

