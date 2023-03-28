# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:40:00 2022

@author: 17000
"""

import torch;
from torch.linalg import norm;
import numpy as np;

class policy_gradient():
    
    def __init__(self, descriptor = 'mtp'):
        if(descriptor == 'mtp'):
            from MTP import moment_tensor_model;
            self.model = moment_tensor_model();
        elif(descriptor == 'deepmd'):
            from Descriptor import descriptor;
            self.model = descriptor();
            
    def set_trajectories(self, trajectory_list):
        self.v, self.atom_index, self.traj, self.R_t = [],[],[],[];
        for i in range(len(trajectory_list)):
            self.v += [a['v'] for a in trajectory_list[i].action];
            self.atom_index += [a['atom_index'] for a in trajectory_list[i].action];
            self.traj += trajectory_list[i].trajectory[:-1];
            self.R_t += [np.sum(trajectory_list[i].reward[t:]) for t in range(len(trajectory_list[i].reward))];
        self.n_frame = len(self.traj);
        self.n_atom = len(self.traj[0].numbers);
        self.sigma = trajectory_list[0].action[0]['sigma'];
        
    def nabla_J(self):
        descriptor = self.model.convert(self.traj);
        v_hat = self.model(descriptor);
        atom_i = torch.tensor([[float(j==self.atom_index[i]) for j in range(self.n_atom)] for i in range(self.n_frame)]);
        v2 = torch.sum(v_hat**2,dim = 2);
        v2i = torch.sum(v2*atom_i,dim=1);
        L1 = torch.log(v2i);
        L2 = -torch.log(torch.sum(v2,dim=1));
        f3 = torch.sum((torch.vstack(self.v)-torch.einsum('ijk,ij->ik',[v_hat,atom_i]))**2, dim=1);
        L3 = -torch.sum(f3*atom_i, dim=1)/2/self.sigma**2;
        L = torch.sum((L1+L2+L3)*torch.tensor(self.R_t))/self.n_frame;
        L.backward();
        return [f.grad for f in self.model.parameters];
    
    
        
        