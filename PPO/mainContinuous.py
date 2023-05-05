




import gym
import numpy as np
from Utils import plot_learning_curve
from Agent_continuos import Agent
#from utils import plot_learning_curve
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    N = 20
    batch_size = 64
    n_epochs = 4
    alpha = 3e-4

    print('---------',env.action_space)
    
    agent = Agent(actions=env.action_space.shape[0], state = np.array(env.reset()[0]).shape, batch_size=batch_size,
                  lr=alpha, epochs=n_epochs,
                  )
    n_games = 10000
    #print('ACTION SPACE0', env.action_space.n)
    figure_file = '/home/fernando/Documents/RemakeCode/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0 
    avg_score = 0
    n_steps = 0
    reward_scl = []
    for i in range(n_games):
        observation = env.reset()
        #rint('----------11', np.shape(observation))
        observation = np.array(observation[0])
        done = False
        score = 0
        reward_scl = []
        while not done:
            observation = np.expand_dims(observation, axis=0)
            action, prob = agent.act(observation)
            #print('----------11', action)
            #env.render()
            action_ = agent.action_limits(action, -1, 1)
            #print('ACTION',i, action)
            observation_, reward, done, info, _ = env.step(action_)

            n_steps += 1
            score += reward
            #action_onehot = np.zeros(4)
            #action_onehot[action] = 1
            #reward_scl.append(reward)
            #reward = agent.reward_scaling(reward_scl)
            agent.store_transition(observation, action,
                                   prob, observation_, reward, done)
            #print('AAAAAAAAaa12', _)
            if n_steps % N == 0:
                #print('ENTROU')
                agent.replay()
            observation = observation_
            if done == True or info == True:
                done = True

        
        

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        #if avg_score > best_score:
         #   best_score = avg_score
          #  agent.save_model(avg_score)

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    end3 = time.time()


    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)



