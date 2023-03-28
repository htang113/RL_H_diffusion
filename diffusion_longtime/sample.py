# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:16:31 2022

@author: 17000
"""

import torch;
from torch.distributions import Categorical;
from torch.distributions.normal import Normal;
from torch.linalg import norm;
import numpy as np;

class policy_gradient():
    
    def __init__(self, model):
        self.model = model;
            
    def set_trajectories(self, trajectory_list):
        self.v, self.atom_index, self.traj, self.R_t = [],[],[],[];
        for i in range(len(trajectory_list)):
            self.v += [torch.tensor(a['v']) for a in trajectory_list[i].action];
            self.atom_index += [a['atom_index'] for a in trajectory_list[i].action];
            self.traj += trajectory_list[i].trajectory[:-1];
#            self.R_t += [np.sum(trajectory_list[i].reward[t:]) for t in range(len(trajectory_list[i].reward))];
            self.R_t += [trajectory_list[i].reward[t] for t in range(len(trajectory_list[i].reward))];
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
        L3 = -f3/2/self.sigma**2/v2i;
        L = torch.sum((L1+L2+L3)*torch.tensor(self.R_t))/self.n_frame;
        L.backward();
        return [f.grad for f in self.model.parameters];
    
    

class sample():
    
    def __init__(self, descriptor = 'mtp',para = [], seed=None):
        if(descriptor == 'mtp'):
            from MTP import moment_tensor_model;
            self.model = moment_tensor_model(parameters = para, seed = seed);
        elif(descriptor == 'deepmd'):
            from Descriptor import descriptor;
            self.model = descriptor();
            self.model.Initialization_para(para);
        self.seed = seed;
    def set_reference(self, conf):
        self.model.set_reference(conf.atoms);
    
    def sample(self, config, T=None, sigma = 0.1):
        
        descriptor = self.model.convert([config]);
        velocity = self.model(descriptor);
        v_abs = norm(velocity[0],dim=1);
        if(T==None):
            partition = v_abs**2;
        else:
            v_abs /= norm(v_abs);
            partition = torch.exp(-v_abs**2/T);
        probability = partition/torch.sum(partition);
        
        ############ ERROR LINE #############
        m_c  = Categorical(probability);
        # The parameter probs has invalid values
        # raise ValueError("The parameter {} has invalid values".format(param))
        
        if(self.seed != None):
            torch.manual_seed(self.seed+1);
            
        self.atom_index = m_c.sample();
        self.sigma = sigma;
        self.p_atom = probability[self.atom_index];
        self.v_mean = velocity[0,self.atom_index];
        m_n = Normal(self.v_mean, sigma*v_abs[self.atom_index]);
        
        if(self.seed != None):
            torch.manual_seed(self.seed+2);
        
        self.v = m_n.sample();
        self.action = [[0]*3]*self.atom_index+[(self.v/norm(self.v)).tolist()]+[[0]*3]*(len(config.numbers)-1-self.atom_index);
        
        return np.array(self.action);
    
    def action_details(self):
        return {'a':self.action, 'v':self.v.tolist(), 'atom_index':int(self.atom_index),
                'p_atom':float(self.p_atom), 'v_mean':self.v_mean.tolist(),'sigma':self.sigma};
    
    def step(self, train_traj,learning_rate = 10**-6):
        from MTP import moment_tensor_model;
        para_old = [torch.tensor(self.model.parameters[i].tolist(),requires_grad=True) for i in range(2)];
        self.model = moment_tensor_model(parameters = para_old);
        optimizer = policy_gradient(self.model);
        optimizer.set_trajectories(train_traj);
        grad = optimizer.nabla_J();
        print(grad)
        para_new = [self.model.parameters[i] + learning_rate*grad[i] for i in range(2)];
        from MTP import moment_tensor_model;
        self.model = moment_tensor_model(parameters = para_new);
        return self.model.parameters;


# =============================================================================
#     
#     def step(self, reward,learning_rate = 10**-3):
#         optimizer = policy_gradient(self.model);
#         optimizer.set_trajectories(train_traj);
#         grad = optimizer.nabla_J();
#         print(grad);
#         para_new = [self.model.parameters[i] + learning_rate*grad[i] for i in range(2)];
#         from MTP import moment_tensor_model;
#         self.model = moment_tensor_model(parameters = para_new);
#         return self.model.parameters;
# =============================================================================
    