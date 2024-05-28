#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:22:58 2024

@author: ubuntu
"""

import json;
from ase import Atoms;
from rlmd.time_model import T_NN;
from torch.optim import SGD, Adam;
from torch.nn import MSELoss;
import torch;
import numpy as np;
from torch.optim.lr_scheduler import StepLR;
from torch.utils.data import DataLoader;
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os;

def example(rank, world_size):

    # create default process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    device = rank;

    with open('../dev/dataset_500_2.json', 'r') as file:
        data_read = json.load(file);

    data_size = 750
    data_select = 750;
    Nepoch = 5000;
    Nstep = 1;
    lr = 1E-4;
    step_size = 50;
    gamma = 1
    tau = 10;

    q_params = {"temperature": 500};

    model1 = T_NN(device, elements = [24,27,28]).to(device);
    optimizer = Adam(model1.parameters(), lr = lr);

    loss_fns = MSELoss();

    with open('loss_'+str(rank)+'.txt', 'w') as file:
        
        file.write('Epoch\t Loss\n');

    np.random.shuffle(data_read);
    data = data_read[data_size*rank : data_size*(rank+1)];
    
    np.random.shuffle(data);
    data = data[:data_select]

    atoms_list = [Atoms(positions = state['state']['positions'],
                    cell = state['state']['cell'],
                    numbers = state['state']['atomic_numbers']) for state in data];

    next_list = [Atoms(positions = state['next']['positions'],
                    cell = state['next']['cell'],
                    numbers = state['next']['atomic_numbers']) for state in data];

    time_list = torch.tensor([state['dt']*(1-state['terminate']) for state in data]).to(device);

    model1.convert(atoms_list + next_list);

    model1 = DDP(model1, device_ids=[rank])
    N_group = int(data_select//Nstep);

    for epoch in range(Nepoch):
        
        record = 0
        optimizer.zero_grad();

        for step in range(Nstep):
            
            indl = torch.tensor([i for i in range(step*N_group, (step+1)*N_group)]).to(device)
            out = model1(indl);
            pred = torch.tanh(out)**2*tau;

            time = time_list[indl];
            
            term1 = tau*(1-torch.exp(-time/tau));
            gamma = torch.exp(-time/tau);
            
            success = (time==0)
                
            label0 = gamma*torch.tanh(model1(indl + data_select))**2*tau + term1;
            label = label0 * (~success);
            
            loss = torch.mean((1 + 1*success) * (pred - label.detach())**2);
        
            record += loss;
            loss.backward();

            if(rank==0 and step==0 and epoch%10==0):
                values = torch.sort(pred).values;
                write_list = [float(values[int(u1)]) for u1 in np.linspace(0, len(values)-1, 6)];
                print(write_list);

            del label
            del loss
            del pred
            torch.cuda.empty_cache()
        
            optimizer.step();
            optimizer.zero_grad();

        if(epoch%10 == 0):
            with open('loss_'+str(rank)+'.txt', 'a') as file:
                file.write(str(epoch)+'\t'+str(float(record/Nstep))+'\n');
            
            if(rank==0):
                torch.save(model1.state_dict(), 'model.pt');

                if(epoch%1000 == 0 and epoch>0):
                    torch.save(model1.state_dict(), 'model_'+str(epoch)+'.pt');

def main():
    world_size = 4;
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    main()