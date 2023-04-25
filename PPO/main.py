




import gym
import numpy as np
from Utils import plot_learning_curve
from Agent import Agent
#from utils import plot_learning_curve
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    env = gym.make('LunarLander-v2', render_mode="rgb_array")
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    #print('---------',type(env.action_space.n))

    agent = Agent(actions=env.action_space.n, state = np.array(env.reset()[0]).shape, batch_size=batch_size,
                  lr=alpha, epochs=n_epochs,
                  )
    n_games = 1000
    print('ACTION SPACE0', env.action_space.n)
    figure_file = '/home/fernando/Documents/RemakeCode/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0 
    avg_score = 0
    n_steps = 0
    for i in range(n_games):
        observation = env.reset()
        
        observation = np.array(observation[0])
        done = False
        score = 0
        while not done:
            #print('----------11', observation)
            action, prob = agent.act(observation)
            env.render()
            observation_, reward, done, info, _ = env.step(action)

            n_steps += 1
            score += reward
            action_onehot = np.zeros(4)
            action_onehot[action] = 1
            
            agent.store_transition(observation, action_onehot,
                                   prob, observation_, reward, done)
            #print('AAAAAAAAaa12', _)
            if n_steps % N == 0:
                a_loss, c_loss = agent.replay()
            observation = observation_

        
        

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



