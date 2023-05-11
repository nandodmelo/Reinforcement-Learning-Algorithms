
from Model import NN
import os
import torch as T
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.distributions.normal import Normal
from Memory import Memory
import gym
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
import random
device = (
    "cuda"
    if T.cuda.is_available()
    else "mps"
    if T.backends.mps.is_available()
    else "cpu"
)
if device != 'cuda':
    print('YOU ARE NOT USING YOUR GPU!!!!!!!!!!!!!!!')

random.seed(1)
np.random.seed(1)
T.manual_seed(1)
T.backends.cudnn.deterministic = True
class PPO:
    def __init__(self, env, gamma = 0.99, lambda_ = 0.95,Entropy_loss = 1e-3, PPO_clip = 0.2, epochs = 10, lr = 3e-4, batch_size = 100000):
        self.env = env
        self.gamma =  gamma
        self.PPO_clip = PPO_clip
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.NN = NN(self.env).to(device)
        self.lamda = lambda_
        self.Entropy_loss = Entropy_loss
        #self.Actor = self.NN.Actor().to(device)
        # print(self.NN.Actor)
        #self.Critic = self.NN.Critic().to(device)
        # print(self.NN.Critic)
        self.scores_ = []
        self.episodes_ = []
        self.average_ = []
        
        self.optimizer = optim.Adam(self.NN.parameters(), lr=3e-4, eps=1e-5)



    # def GAE(self, rewards, next_values, values, dones, normalize = True):
    #     #https://arxiv.org/pdf/1506.02438.pdf - Eq 17
    #     #SUMMATION A = r(t) - V(st) + gamma**(k-1)*r(t+k-1)+gamma**(k)*V(t+k)
    #     with T.no_grad():
    #         adv = T.zeros(self.batch_size,).to(device)
    #         returns = T.zeros(self.batch_size,).to(device)
    #         #print('----', np.shape(next_values))
    #         deltas = [r + self.gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
    #         #print('ADv', len(deltas))
    #         for t in reversed(range(len(deltas) - 1)):
    #             adv[t] = deltas[t] + (1-dones[t]*self.gamma * self.lamda * deltas[t+1])

    #         if normalize == True:
    #             adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    #         returns = adv - values
    #         return adv, returns
    def GAE(self, rewards, next_values, values, dones, normalize = True):
        with T.no_grad():
            
            #print('SHAPE', rewards.size(), next_values.size(), values.size(), dones.size())
            advantages = T.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(2048)):
                if t == 2048 - 1:
                    nextnonterminal = 1.0 - dones[-1]
                    nextvalues = next_values[-1]
                else:


                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]

                advantages[t] = lastgaelam = delta + self.gamma * self.lamda * nextnonterminal * lastgaelam
            returns = advantages + values        
            return advantages, returns
        
    def act(self, x, action = None):
        action_mean = self.NN.Actor(x)
        
        action_logstd = self.NN.Actor_logstd.expand_as(action_mean)
        action_std = T.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.NN.Critic(x)


    def replay(self, rewards, next_states, states, dones, probs, actions, values):
        indexs = np.arange(self.batch_size)
        #print('indexes', np.shape(rewards))
        with T.no_grad():
            
            next_value = self.NN.get_value(next_states)
            #print('AQUI', np.shape(next_value))
            adv, returns = self.GAE(rewards, T.squeeze(next_value), values, dones)
        
        returns =  returns.reshape(-1)
        adv =  adv.reshape(-1)
        probs = probs.reshape(-1)
        #print('action', np.shape(actions), 'returns', np.shape(returns), 'adv', np.shape(adv))
        #print('states', np.shape(states), 'nextstates', np.shape(next_states), 'probs', np.shape(probs))
        for _ in range(self.epochs):
            np.random.shuffle(indexs)

            for batch in range(0, self.batch_size, 64):

                end = batch + 64
                ind = indexs[batch:end]
                
                #print('BATCH', a)
                old_prob = probs[ind]
                states_aux = states[ind]
                value_aux = values[ind]
                returns_aux = returns[ind]
                adv_aux = adv[ind]  
                _, new_prob, entropy, critic_value_ = self.act(states_aux, actions[ind])
                #print('actions2', _)
                  
                ratio = T.exp(new_prob - old_prob)

                p1 = -adv_aux * ratio
                p2 = -adv_aux * T.clamp(ratio, 1- self.PPO_clip, 1 + self.PPO_clip)
                #print('SHAPES', np.shape(p1), np.shape(p2))
                actor_loss = T.max(p1,p2).mean()
                entropy_loss = entropy.mean()
                
                
                #CriticLoss
                #value_aux = value[ind]
                
                #critic_value = self.NN.Critic(states_aux)  
                critic_value = critic_value_.view(-1)
                #print('LL0',np.shape(critic_value), 'LL0',np.shape(critic_value_))
                
                
            
                
                critic_clipped = value_aux + T.clamp((critic_value - T.squeeze(value_aux)), 1- self.PPO_clip, 1 + self.PPO_clip)
                critic_loss_clipped = (critic_clipped - returns_aux) ** 2
                
                critic_loss_noclip = (critic_value - returns_aux)**2
                critic_loss_max = T.max(critic_loss_noclip, critic_loss_clipped)
                critic_loss = 0.5 * critic_loss_max.mean()
                #critic_loss = 0.5 * ((critic_value - returns_aux) ** 2).mean()
                #print('LOSS',np.shape(critic_loss),np.shape(actor_loss), np.shape(entropy_loss) )
                loss = actor_loss - entropy_loss* self.Entropy_loss + critic_loss* 0.5
                #print('LOSSFINAL', actor_loss, entropy_loss, critic_loss, loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.NN.parameters(), 0.5)
                self.optimizer.step()







    def action_clip(self, action, action_space):
        return np.clip(action, action_space.low, action_space.high)


    def PlotModel(self, score, episode):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        return self.average_[-1]


if __name__ == "__main__":
    env_id = 'BipedalWalker-v3'
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env.seed(1)
    env.action_space.seed(1)
    env.observation_space.seed(1)
    batch_size = 2048
    Agent = PPO(env, batch_size = batch_size)

    action_size = env.action_space.shape[0]

    episodes = 0
    score = 0
    while True:
        states = T.zeros(batch_size, 24).to(device)
        actions = T.zeros(batch_size, env.action_space.shape[0]).to(device)
        probs = T.zeros(batch_size).to(device)
        next_states = T.zeros(batch_size, 24).to(device)
        rewards = T.zeros(batch_size).to(device)
        values = T.zeros(batch_size).to(device)
        dones = T.zeros(batch_size).to(device)
        
        obs = T.tensor(np.expand_dims(env.reset(), axis=0), dtype=T.float32).to(device)
        #print('firstobs', obs)
        # this is each frame, up to 500...but we wont make it that far with random.
        mean_reward = []
        for i in range(2048):
            
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            #env.render()
            
            # This will just create a sample action in any environment.
            # In this environment, the action can be any of one how in list on 4, for example [0 1 0 0]
            
            with T.no_grad():
                action, prob, entropy, value = Agent.act(obs)
                value = value.flatten()
            #print('firstobs',action, prob, entropy, value)
            #print('shape', np.shape(action.cpu().numpy()))
            # this executes the environment with an action, 
            # and returns the observation of the environment, 
            # the reward, if the env is over, and other info.
            # action = action_clip(np.squeeze(action.cpu().numpy()), env.action_space)
            #print('ACTION',action, np.shape(action))
            next_state, reward, done, info = env.step(np.squeeze(action.cpu().numpy()))
            #print('probs', next_states.size(), states.size())
            probs[i] = prob
            actions[i:] = T.squeeze(action[0])
            next_states[i:] = T.from_numpy(next_state)
            states[i:] = obs
            values[i] = value
            rewards[i] = T.tensor(reward).view(-1)
            
            dones[i] = done

            obs = T.unsqueeze(T.Tensor(next_state).to(device), dim=0)
            #print(obs.size())
            score += reward

            if done:
                episodes += 1
                average = Agent.PlotModel(score, episodes)
                print("episode: {}/{}, score: {}, average: {:.2f}".format(episodes, 10000, info['episode']['r'], average))
                state, done, score= T.tensor(np.expand_dims(env.reset(), axis=0), dtype=T.float32).to(device), False, 0
        Agent.replay(rewards, next_states, states, dones, probs, actions, values)
        if episodes >= 10000:
            break


    plt.plot(mean_reward)
    
    
    plt.draw()
    plt.savefig('bipedal.png')
    plt.show()