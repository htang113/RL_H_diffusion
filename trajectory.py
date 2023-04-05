# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:10:21 2022

@author: 17000
"""

import ase;
import numpy as np;
import scipy;
from itertools import chain;
from scipy.sparse import linalg;
from ase import io;

class trajectory(object):
    
    # This class records the trajectory and critical coefficients along the trajectory
    # Alternatively apply add_minimum and add_saddle, passing 'configuration' objects to the methods
    # Trajectories can be output as lammps output files, which can be visualized by OVITO or AtomEye
      
    def __init__(self, k_s, k_min, k_iter=0):
        self.k_s          = k_s;
        self.k_min        = k_min;
        self.states       = [];
        self.next_states  = [];
        self.actions      = [];
        self.act_space    = [];
        self.action_probs = [];
        self.rewards      = [];
        
        self.E_min      = [];
        self.E_s        = [];
        self.E_next     = [];
        self.fail       = [];
        
        self.fail_panelty = -5;
        self.kb = 1.380649/1.602*10**-4;
        self.meV_to_Hz = 1.602/6.626*10**12;
        
    def add(self, info):
        self.states.append(info['state']);
        self.actions.append(info['act']);
        self.act_space.append(info['act_space']);
        self.action_probs.append(info['act_probs']);
        self.fail.append(info['fail']);
        self.next_states.append(info['next']);
        
        self.E_min.append(info['E_min']);
        self.E_next.append(info['E_next']);
        
        if(not info['fail']):
            self.E_s.append(info['E_s']);
            self.rewards.append(self.k_s*(self.E_min[-1]-self.E_s[-1])+self.k_min*(self.E_min[-1]-self.E_next[-1]));
        else:
            self.E_s.append(None);
            self.rewards.append(self.fail_panelty);
    
    def HTST(self, T):
        self.t_list = [];
        for i in range(len(self.E_s)):
            if(type(self.freq_s[i]) == type(None) or type(self.freq_min[i]) == type(None)):
                raise('Error: frequency has not been calculated, so time cannot be evaluated.')
            f_s = np.log(np.sqrt(np.sort(self.freq_s[i][1:])));
            f_m = np.log(self.freq_min[i]);
            exp_term  = np.exp(-(self.E_s[i]-self.E_min[i])/self.kb/T);
            freq_term = np.exp(np.sum(f_m)-np.sum(f_s))/2*np.pi*self.meV_to_Hz;
            self.t_list.append(1/(exp_term*freq_term));
        return self.t_list;
    
    def to_file(self, filename, animation=False):
        traj_list = [];
        for i in range(len(self.E_s)):
            traj_list.append(self.trajectory[i]);
            traj_list.append(self.saddle[i]);
        traj_list.append(self.trajectory[-1]);
        io.write(filename, traj_list, format='vasp-xdatcar');
        if(animation):
            io.write(filename, traj_list, format='mp4');  

    def save(self, filename):
        to_list = [self.k_s,
        self.k_min,
        [{key: u.todict()[key].tolist() for key in u.todict()} for u in self.trajectory],
        self.E_min,
        [{key: u.todict()[key].tolist() for key in u.todict()} for u in self.saddle],
        self.E_s,
        self.freq_min,
        self.freq_s,
        self.fail,
        self.reward,
        self.action,
        self.iter_search,
        self.k_iter,
        self.kb,
        self.meV_to_Hz];
        import json;
        with open(filename,'w') as file:
            json.dump(to_list,file);
        return to_list;
    
    def load(self, filename):
        import json;
        with open(filename,'r') as file:
            data = json.load(file);
        [self.k_s,
        self.k_min,
        trajectory,
        self.E_min,
        self.saddle,
        self.E_s,
        self.freq_min,
        self.freq_s,
        self.fail,
        self.reward,
        self.action,
        self.iter_search,
        self.k_iter,
        self.kb,
        self.meV_to_Hz] = data;
        self.trajectory = [];
        for atoms in trajectory[:-1]:
            self.trajectory.append(ase.Atoms.fromdict(atoms));
        self.action = self.action[:-1];
        return data;
        
        