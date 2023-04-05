# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:40:00 2022

@author: 17000
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from model import descriptorNN;

class PPO:
    def __init__(self, model, lr=10**-3, eps_clip=0.01, temperature = 300):
        self.policy_value_net = model;
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr)
        self.eps_clip = eps_clip;
        self.kT = temperature*8.617*10**-5;

    def select_action(self, state, action_space):
        policy, _ = self.policy_value_net(state,action_space);
        action_probs = torch.exp(policy);
        action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy())
        return action, action_probs;

    def update(self, memory, gamma):
        states = memory.states
        aspace = memory.act_space;
        actions = memory.actions;
        old_action_probs = memory.action_probs
        rewards = torch.tensor(memory.rewards, dtype = torch.float)
        next_states = memory.next_states;

        values = [float(self.policy_value_net(states[i],aspace[i])[1]) for i in range(len(states))]
        next_values = torch.tensor(values[1:]+[float(self.policy_value_net(next_states[-1],aspace[-1])[1])])
        values = torch.tensor(values);
        
        returns = rewards + gamma * next_values
        advantages = returns - values

        for _ in range(10):
            for i in range(len(states)):
                policy, value = self.policy_value_net(states[i],aspace[i]);
                dist = torch.distributions.Categorical(torch.exp(policy));
                entropy = dist.entropy();
                
                ratio = torch.exp(policy[actions[i]])/old_action_probs[i][actions[i]];
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[i]
                if(i==0):
                    loss = -torch.min(surr1, surr2);
                else:
                    loss += -torch.min(surr1, surr2);
                loss += nn.MSELoss()(returns[i], value)

                loss -= self.kT * entropy;
                
            loss = loss/len(states);
            self.optimizer.zero_grad();
            loss.backward();
            self.optimizer.step();
            
        return loss;
         