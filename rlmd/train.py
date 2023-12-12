# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:40:00 2022

@author: 17000
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rlmd.model import DQN;

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
        states = memory.states;  
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

class Context_Bandit:
    def __init__(self, model, lr=10**-3, eps_clip=0.01, temperature = 300):
        self.policy_value_net = model;
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr)
        self.eps_clip = eps_clip;
        self.kT = temperature*8.617*10**-5;

    def select_action(self, state, action_space):
        self.policy_value_net.convert([state],[action_space]);
        Q = self.policy_value_net()[0];
        action_probs = nn.Softmax(dim=0)(Q[:,0]/self.kT+Q[:,1]);
        action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy());
        
        return action, action_probs, Q;

    def update(self, memory_l, gamma, steps):
        ave_loss = 0;
        for m in range(steps[0]):
            prob = [0.99**(len(memory_l)-i) for i in range(len(memory_l))];
            randint = np.random.choice(range(len(memory_l)), p=prob/np.sum(prob));
            memory = memory_l[randint];
            states = memory.states
            aspace = memory.act_space;
            actions = memory.actions;
            taken_actions = [[aspace[i][actions[i]]] for i in range(len(aspace))];
            rewards = torch.tensor(memory.rewards, dtype = torch.float)
            freq   = torch.tensor(memory.freq, dtype = torch.float)
            self.policy_value_net.convert(states,taken_actions);
            for epoch in range(steps[1]):
                Q = self.policy_value_net();
                loss = torch.mean((rewards-Q[:,0,0])**2+(freq-Q[:,0,1])**2*self.kT**2);
                self.optimizer.zero_grad();
                loss.backward();
                self.optimizer.step();
                ave_loss += loss;
        return ave_loss/steps[0]/steps[1];
    

class Q_trainer:
    def __init__(self, model, target, lr=10**-3, temperature = 300):
        self.policy_value_net = model;
        self.target_net = target;
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr)
        self.kT = temperature*8.617*10**-5;

    def select_action(self, state, action_space):
        self.policy_value_net.convert([state],[action_space]);
        Q = self.policy_value_net()[0];
        action_probs = nn.Softmax(dim=0)(Q[:,0]/self.kT);
        action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy());
        
        return action, action_probs, Q;

    def update(self, memory_l, gamma, steps, size=10): 
        self.target_net.load_state_dict(self.policy_value_net.state_dict());
        ave_loss = 0;
        for m in range(steps[0]):
            randint = np.random.randint(len(memory_l),size = size);
            states, next_states, taken_actions, rewards, next_aspace = [],[],[],[],[];
            for u in randint:
                memory = memory_l[u];
                states += memory.states[:-1];
                next_states += memory.next_states[:-1];
                rewards += memory.rewards[:-1];
                aspace = memory.act_space;
                actions = memory.actions;    
                taken_actions += [[aspace[i][actions[i]]] for i in range(len(aspace)-1)];
                next_aspace += [aspace[i] for i in range(1,len(aspace))];
            rewards = torch.tensor(rewards, dtype = torch.float);
            self.policy_value_net.convert(states, taken_actions);
            next_Q = torch.zeros(len(next_aspace));
            for i in range(len(next_aspace)):
                self.target_net.convert([next_states[i]],  [next_aspace[i]])
                next_Q[i] = torch.max(self.target_net());
            
            for epoch in range(steps[1]):
                Q = self.policy_value_net();
                target  = next_Q*gamma+rewards;
                loss = torch.mean((target-Q[:,0,0])**2);

                self.optimizer.zero_grad();
                loss.backward(retain_graph=True);
                self.optimizer.step();
                ave_loss += loss;
                
        return ave_loss/steps[0]/steps[1];