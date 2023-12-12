from rlmd.configuration import configuration;
from rlmd.trajectory import trajectory;
from rlmd.step import environment;
from rlmd.model import DQN;
from rlmd.train import Q_trainer;
from rlmd.action_space import actions;

import numpy as np;
import json;
import torch;
import os;

task = 'Cu_DQN/H3';
horizon = 30;
n_traj = 300;

model = DQN(elements=[1,29],r_cut = 5, 
             N_emb=24, N_fit = 128, atom_max = 60);
target = DQN(elements=[1,29],r_cut = 5, 
             N_emb=24, N_fit = 128, atom_max = 60);
#model.load('Cu_all/model/model90')
trainer = Q_trainer(model, target, lr=10**-4, temperature = 1000);

pool = ['POSCARs/H/CONTCAR_H_111Cu_H'+str(i) for i in range(10)];

traj_list = [];
if(task not in os.listdir()):
    os.mkdir(task);
if('traj' not in os.listdir(task)):
    os.mkdir(task+'/traj');
if('model' not in os.listdir(task)):
    os.mkdir(task+'/model');

with open(task+'/loss.txt','w') as file:
    file.write('epoch\t loss\n');

for epoch in range(n_traj):
    trainer.optimizer.lr = 10**-3*0.99**epoch;
    conf = configuration();
    file = pool[np.random.randint(len(pool))];
    conf.load(file);
    print('epoch = '+str(epoch)+':  '+file);
    conf.set_potential();
    env = environment(conf, logfile = task+'/log', max_iter=100);
    env.relax(accuracy = 0.1)
    traj_list.append(trajectory(0,1));
    for tstep in range(horizon):
        
        action_space = actions(conf,dist_mul_body = 1.2, act_mul = 1.6,act_mul_move = 1.2,collision_r = 0.5);
        act_id, act_probs,Q = trainer.select_action(conf.atoms,action_space);
        action = action_space[act_id];
        info = {'act':act_id, 'act_probs':act_probs.tolist(),'act_space':action_space,'state':conf.atoms.copy(),
                'E_min':conf.potential()};
        
        E_next, fail = env.step(action, accuracy = 0.05);
        env.normalize();
        
        info['next'], info['fail'], info['E_next'] =conf.atoms.copy(), fail, E_next;
        info['E_s'], info['log_freq'] = 0, 0;
        traj_list[-1].add(info);
        if(fail):
            print('fail');
            
    loss = trainer.update(traj_list, 0.8, [int(1+epoch**(2/3)),5],size=1);
    with open(task+'/loss.txt','a') as file:
        file.write(str(epoch)+'\t'+ str(loss)+'\n');
    try:
        traj_list[epoch].save(task+'/traj/traj'+str(epoch));
    except:
        print('saving failure');
        
    if(epoch%10==0):
        model.save(task+'/model/model'+str(epoch));
    
