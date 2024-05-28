#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:18:49 2024

@author: ubuntu
"""

import json;
from ase import Atoms;
from ase import io;

from rlmd.time_model import T_NN;
from torch.optim import SGD;
from torch.nn import MSELoss;
import torch;
import numpy as np;
from torch.optim.lr_scheduler import StepLR;

device = 'cuda:0'

data_size = 500
Nepoch = 1000;
Nstep = 3;
lr = 1E-4;
step_size = 50;
gamma = 0.9
tau = 10;

vacancy = 2;
q_params = {"temperature": 500};
T = q_params['temperature'];

model = T_NN(device, elements = [24,27,28]).to(device);

filename = 'models/model_'+str(T)+'_' +str(vacancy)+ '.pt'
try:
    model.load_state_dict(torch.load(filename));
except:
    res = torch.load(filename);
    for key in list(res.keys()):
        res[key[7:]] = res[key];
        del res[key];
    model.load_state_dict(res);

atoms_l = [io.read('POSCARs/DQN_'+ str(256-vacancy) + '_'+ str(T) + 'K/XDATCAR'+str(k), index=':') for k in range(10)];
Nframe = len(atoms_l);3

out = [];

#time_list = [0]*Nframe;

#graph = make_p_dataset(atoms_list, time_list,  q_params, 5).data_list;
#graph = [u.to(device) for u in graph];

out = [];


for j in range(200):
    i = 10*j;
    out.append([]);
    for k in range(10):
        model.convert([atoms_l[k][i]]);    
        pred = torch.tanh(model(torch.tensor([0]).to(device)))**2*tau;
    
        out[-1] += [float(pred)];
    if(i%20==0):
        print('complete '+str(i//20+1)+'%');
        
with open('time_map_'+str(T)+'_' +str(vacancy)+ '.json','w') as file:
    json.dump(out,file);
