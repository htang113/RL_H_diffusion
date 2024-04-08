#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:18:49 2024

@author: ubuntu
"""

import json;
from ase import Atoms;
from ase import io;
from rlmd.utils import make_p_dataset;
from rlmd.time_model import PNet2;
from torch.optim import SGD;
from torch.nn import MSELoss;
import torch;
import numpy as np;
from torch.optim.lr_scheduler import StepLR;

device = 'cpu'

data_size = 500
Nepoch = 1000;
Nstep = 3;
lr = 1E-4;
step_size = 50;
gamma = 0.9
tau = 1;

q_params = {"temperature": 500};

model = PNet2([24,27,28]).to(device);
model.load_state_dict(torch.load('model.pt'));

with open('POSCARs/random_SRO/record.json', 'r') as file:
    
    SRO = json.load(file);

out = [];
for i in range(500):
    atoms_list = [io.read('POSCARs/random_SRO/POSCAR_'+str(i))];
    time_list = [0];
    
    graph = make_p_dataset(atoms_list, time_list,  q_params, 5).data_list;
    graph = [u.to(device) for u in graph];

    pred = model(graph[0]);

    out += [[SRO[i], float(pred)]];
with open('time_map.json','w') as file:
    json.dump(out,file);
