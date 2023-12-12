from rlmd.configuration import configuration;
from rlmd.trajectory import trajectory;
from rlmd.step import environment;
import numpy as np;
import multiprocessing;
import json;

from rlmd.model import Q_NN;
import torch;
from rlmd.train import Context_Bandit;
from rlmd.action_space import actions;
import os;
from ase import io;

sro = '0.8'
T = 400;
kT =  T*8.617*10**-5;

horizon = 500;
model = Q_NN(elements=[1,24,27,28],r_cut = 5, 
             N_emb=24, N_fit = 128, atom_max = 50);
model.load('model_context_bandit.pt');
trainer = Context_Bandit(model, temperature = T);

Tl = [];
Cl = [];
for u in range(50,100):
    conf = configuration();
    conf.load('Fig3/'+sro+'/POSCAR_'+str(u));
    conf.set_potential();
    env = environment(conf, max_iter=100);
    env.relax(accuracy = 0.1);
    
    filename = 'Fig3/'+sro+'/XDATCAR'+str(u);
    io.write(filename, conf.atoms, format='vasp-xdatcar');
    tlist = [0];
    clist = [conf.atoms.get_positions()[-1].tolist()];
    for tstep in range(horizon):
        action_space = actions(conf);
        act_id, act_probs,Q = trainer.select_action(conf.atoms,action_space);
        Gamma = float(torch.sum(torch.exp(Q[:,0]/kT+Q[:,1])));
        dt   = 1/Gamma*10**-6;
        tlist.append(tlist[-1]+dt);
        action = action_space[act_id];
        E_next, fail = env.step(action, accuracy = 0.1);
        io.write(filename, conf.atoms, format='vasp-xdatcar',append = True);
        clist.append(conf.atoms.get_positions()[-1].tolist());
        if(tstep%100==0):
            print(str(u)+': '+str(tstep));
    Tl.append(tlist);
    Cl.append(clist);
    with open('Fig3/'+sro+'/diffuse.json','w') as file:
        json.dump([Tl,Cl], file);
    
