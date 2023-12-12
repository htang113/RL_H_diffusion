from rlmd.configuration import configuration;
from rlmd.trajectory import trajectory;
from rlmd.step import environment;
import numpy as np;
import multiprocessing;
import json;

from rlmd.model import DQN2;
from rlmd.train import Q_trainer;
import torch;
from rlmd.action_space import actions;
import os;
from ase import io;

horizon = 200;
model = DQN2(elements=[1,29],r_cut = 8.5, 
             N_emb=24, N_fit = 128, atom_max = 260);
target = DQN2(elements=[1,29],r_cut = 8.5, 
             N_emb=24, N_fit = 128, atom_max = 260);
model.load('model_DQN.pt');
target.load('model_DQN.pt');
trainer = Q_trainer(model, target, lr=10**-4, temperature = 100);

El = [];
for u in range(50):
    conf = configuration();
    conf.load('POSCARs/pos/POSCAR'+str(u));
    conf.set_potential();
    env = environment(conf, max_iter=100);
    env.relax(accuracy = 0.1);
    
    filename = 'POSCARs/pos/XDATCAR'+str(u);
    io.write(filename, conf.atoms, format='vasp-xdatcar');
    Elist = [conf.potential()];
    for tstep in range(horizon):
        T = 1000-950*tstep/(horizon-1);
        trainer.kT = T*8.617*10**-5;
        action_space = actions(conf);
        act_id, act_probs,Q = trainer.select_action(conf.atoms,action_space);
        action = action_space[act_id];
        E_next, fail = env.step(action, accuracy = 0.1);
        io.write(filename, conf.atoms, format='vasp-xdatcar',append = True);
        Elist.append(conf.potential());
        if(tstep%100==0):
            print(str(u)+': '+str(tstep));
    El.append(Elist);
    with open('POSCARs/pos/converge.json','w') as file:
        json.dump(El, file);
    
