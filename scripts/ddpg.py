from hashlib import new
import numpy as np
import csv
from pathlib import Path
import rospy

import torch
import torch.nn as nn
from torch.optim import Adam

from torch import *
from torch.autograd import Variable
import torch.nn as nn
from networks import *
from replay_buffer import *
import os

#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))
logPath = dirPath.replace("/scripts", "/log")
dirPath = dirPath.replace("/scripts", "/models")


class DDPGagent:
    def __init__(self, space_states_dim, space_actions_dim, buffer_size, gamma= 0.99, alphaActor=1e-4, alphaCritic=1e-3, tau= 1e-3):
        # Params
        self.tau = tau
        self.gamma= gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.save_critic_loss = None
        self.save_policy_loss = None
        self.save_Qvals = None
        self.episode= 0

        # Networks
        self.actor = Actor(space_states_dim, space_actions_dim, device= self.device)
        self.actor_target = Actor(space_states_dim, space_actions_dim, device= self.device)
        


        self.critic = Critic(space_states_dim, space_actions_dim, device= self.device)
        self.critic_target = Critic(space_states_dim, space_actions_dim, device= self.device)
        
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = ReplayBuffer(buffer_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = Adam(self.actor.parameters(), lr=alphaActor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=alphaCritic)

        
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor.forward(state).to(self.device)
        return action.detach().cpu().numpy()

    
    
    def update(self, batch_size):
        states, actions, rewards, next_states, done = self.memory.sample(batch_size)

        

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
    
        # Critic loss        
        

        next_actions = self.actor_target.forward(next_states).detach()
        next_Q = self.critic_target.forward(next_states, next_actions).squeeze(1).detach()

        Qprime = rewards + ((1 - done) * self.gamma * next_Q)

        Qvals = self.critic.forward(states, actions).squeeze(1)

        

        critic_loss = self.critic_criterion(Qvals, Qprime)

        self.save_critic_loss = critic_loss.item()

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        self.save_policy_loss = policy_loss.item()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_models(self, episode_count):
        if episode_count%20 == 0:
            torch.save(self.actor_target.state_dict(), dirPath +"/"+ str(episode_count)+ '_actor.pt')
            torch.save(self.critic_target.state_dict(), dirPath +"/"+ str(episode_count)+ '_critic.pt')
            print('****Models saved***')
        
    def load_models(self, episode):
        self.actor.load_state_dict(torch.load(dirPath +"/"+ str(episode)+ '_actor.pt'))
        self.critic.load_state_dict(torch.load(dirPath +"/"+ str(episode)+ '_critic.pt'))

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        print('***Models load***')

    def save_rewards(self, episode, reward):
        
        f = open(logPath +"/log.csv", "a", encoding='utf-8', newline='')
            
        w = csv.writer(f)
        w.writerow([episode, reward, self.save_Qvals, self.save_critic_loss, self.save_policy_loss])
        f.close() 
        print('****Rewards saved***')

    def save_state(self, episode, step, reward, state):
        if episode%10 == 0:
            self.episode = episode
        
        f = open(logPath +"/observation/"+str(self.episode)+"_state.csv", "a", encoding='utf-8', newline='')
            
        w = csv.writer(f)
        w.writerow([episode, step, reward, [state[0], state[1]], [state[2], state[3]]])
        f.close() 
        print('****Observation saved***')

    def save_test(self, test, episode, reward, time, vel):
        f = open(logPath +"/test/test.csv", "a", encoding='utf-8', newline='')
        w = csv.writer(f)
        w.writerow([test, episode, reward, time, vel[0], vel[1]])
        f.close() 
        
        print('****Test saved***')

    

            

        
