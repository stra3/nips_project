from controllers.neuralqcontroller import Neuralqcontroller
from controllers.qcontroller import Qcontroller
from controllers.controller import Controller
from env import Env
import gym
import numpy as np

if __name__ == '__main__':
    
    env = Env('CartPole-v1')
    control = Neuralqcontroller(env)
    render = True
    n_episodes = 1000 

    for e in range(n_episodes):
        obs = env.reset()
        done = False

        while done==False:
            if render == True:
                env.render()
            action = control.generateAction(obs, e)
            obs, reward ,done,_ = env.step(action)
            # control.updateQtable(e, reward, obs)

    print("----Finished Run----")
