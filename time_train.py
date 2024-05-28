#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:22:58 2024

@author: ubuntu
"""

import json;
from ase import Atoms;
from rlmd.utils import make_p_dataset;
from rlmd.time_model import T_NN;
from torch.optim import SGD, Adam;
from torch.nn import MSELoss;
import torch;
import numpy as np;
from torch.optim.lr_scheduler import StepLR;
from torch.utils.data import DataLoader;

device = 'cuda:0'

with open('dev/dataset_500.json', 'r') as file:
    data_read = json.load(file);

data_size = 1000
Nepoch = 5000;
Nstep = 1;
lr = 1E-3;
step_size = 50;
gamma = 1
tau = 1;

q_params = {"temperature": 500};

model1 = T_NN(device, elements = [24,27,28]).to(device);
#model2 = T_NN(device, elements = [24,27,28]).to(device);
#model1.load_state_dict(torch.load('model2.pt'));

optimizer = Adam(model1.parameters(), lr = lr);
#scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma);
loss_fns = MSELoss();

with open('loss_2.txt', 'w') as file:
    
    file.write('Epoch\t Loss\n');
np.random.shuffle(data_read);
data = data_read[:data_size];

atoms_list = [Atoms(positions = state['state']['positions'],
                cell = state['state']['cell'],
                numbers = state['state']['atomic_numbers']) for state in data];

next_list = [Atoms(positions = state['next']['positions'],
                cell = state['next']['cell'],
                numbers = state['next']['atomic_numbers']) for state in data];

time_list = torch.tensor([state['dt']*(1-state['terminate']) for state in data]).to(device);

model1.convert(atoms_list + next_list);
#model2.convert(next_list);

N_group = int(data_size//Nstep);

for epoch in range(Nepoch):
    
    record = 0
    optimizer.zero_grad();

    for step in range(Nstep):
            
#        graph_in = [u.to(device) for u in graph[step*N_group:(step+1)*N_group]];
#        label_in = [u.to(device) for u in labels[step*N_group:(step+1)*N_group]];
#        for i in range(step*N_group, (step+1)*N_group):
        indl = torch.tensor([i for i in range(step*N_group, (step+1)*N_group)]).to(device)
        pred = model1(indl)*tau;

        time = time_list[indl];
        
        term1 = tau*(1-torch.exp(-time/tau));
        gamma = torch.exp(-time/tau);
        
        success = (time==0)
            
        label0 = gamma*model1(indl + data_size)*tau + term1;
        label = label0 * (~success);
        
        loss = torch.mean((1 + 0*success) * (pred - label.detach())**2);
    
        record += loss;
        loss.backward();

        del label
        del loss
        del pred
        torch.cuda.empty_cache()
    
        optimizer.step();
        optimizer.zero_grad();
#    scheduler.step();
        
    with open('loss_2.txt', 'a') as file:
        file.write(str(epoch)+'\t'+str(float(record/Nstep))+'\n');
    
    torch.save(model1.state_dict(), 'model2.pt');
#    if(epoch%10 == 9):
#        model2.load_state_dict(model1.state_dict());
    if(epoch%20 == 0):
        print('Epoch ' + str(epoch) + ', loss: ' + str(float(record/Nstep)));

#with open('pred.json','w') as file:
#    json.dump(pred.tolist(), file);
