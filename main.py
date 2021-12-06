from controllers.qcontroller import Qcontroller
from controllers.controller import Controller
from env import Env
import gym
import numpy as np

if __name__ == '__main__':
    
    env = Env('CartPole-v1')
    control = Qcontroller(env)

    n_episodes = 10000 
    for e in range(n_episodes):
        obs = env.reset()
        done = False
        # print(control.Q_table)

        while done==False:
            if e%100 == 0:
                print(e)
            if e>1000:
                env.render()
            action = control.generateAction(obs, e)
            obs, reward ,done,_ = env.step(action)
            control.updateQtable(e, reward, obs)
