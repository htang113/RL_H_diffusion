#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:22:58 2024

@author: ubuntu
"""

import json;
from ase import Atoms;
from rlmd.utils import make_p_dataset;
from rlmd.time_model import PNet2;
from torch.optim import SGD;
from torch.nn import MSELoss;
import torch;
import numpy as np;
from torch.optim.lr_scheduler import StepLR;

device = 'cpu'

with open('dev/dataset.json', 'r') as file:
    data_read = json.load(file);

data_size = 500
Nepoch = 1000;
Nstep = 3;
lr = 1E-4;
step_size = 50;
gamma = 0.9
tau = 1;

q_params = {"temperature": 500};

model1 = PNet2([24,27,28]).to(device);
model2 = PNet2([24,27,28]).to(device);
model2.load_state_dict(model1.state_dict());

optimizer = SGD(model1.parameters(), lr = lr);
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma);
loss_fns = MSELoss();

with open('loss_time.txt', 'w') as file:
    
    file.write('Epoch\t Loss\n');

for epoch in range(Nepoch):
    
    np.random.shuffle(data_read);
    data = data_read[:data_size];

    atoms_list = [Atoms(positions = state['state']['positions'],
                  cell = state['state']['cell'],
                  numbers = state['state']['atomic_numbers']) for state in data];

    next_list = [Atoms(positions = state['next']['positions'],
                  cell = state['next']['cell'],
                  numbers = state['next']['atomic_numbers']) for state in data];

    time_list = [state['dt']*(1-state['terminate']) for state in data];
    graph = make_p_dataset(atoms_list, time_list,  q_params, 5).data_list;
    labels = make_p_dataset(next_list, time_list,  q_params, 5).data_list;
    graph = [u.to(device) for u in graph];
    labels = [u.to(device) for u in labels];
    
    record = 0
    for step in range(Nstep):
        optimizer.zero_grad();
            
        for i in range(len(graph)):
            
            pred = model1(graph[i]);
            time = labels[i].time;
            
            term1 = tau*(1-torch.exp(-time/tau));
            gamma = np.exp(-time/tau);
            
            if(time == 0):
                
                loss = loss_fns(pred, torch.tensor(0., dtype=torch.float));
                
                record += loss;
                loss.backward();
            
            else:
                
                label = gamma*model2(labels[i]).detach() + term1;
                loss = loss_fns(pred, label);
        
                record += loss;
                loss.backward();
            
        optimizer.step();
    
    scheduler.step();
    
    print('Epoch ' + str(epoch) + ', loss: ' + str(float(record/len(graph)/Nstep)));
    
    with open('loss_time.txt', 'a') as file:
        file.write(str(epoch)+'\t'+str(float(record/len(graph)/Nstep))+'\n');
    
    torch.save(model1.state_dict(), 'model.pt');
    model2.load_state_dict(model1.state_dict());

