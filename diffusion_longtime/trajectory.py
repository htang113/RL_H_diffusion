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
        self.k_s = k_s;
        self.k_min = k_min;
        self.trajectory = [];
        self.E_min      = [];
        self.saddle     = [];
        self.E_s        = [];
        self.freq_min   = [];
        self.freq_s     = [];
        self.fail       = [];
        self.reward     = [];
        self.action     = [];
        self.iter_search= [];
        self.k_iter = k_iter;
        self.kb = 1.380649/1.602*10**-4;
        self.meV_to_Hz = 1.602/6.626*10**12;
        
    def add_minimum(self, config, calc_Hessian = False):
        if(self.fail != [] and self.fail[-1]):
            self.trajectory.append(self.trajectory[-1]);
            self.E_min.append(self.E_min[-1]);
            self.freq_min.append((self.freq_min[-1]));
        else:
            self.trajectory.append(config.atoms);
            self.E_min.append(config.potential());
            if('omega2' in dir(config)):
                self.freq_min.append(np.sqrt(config.omega2).tolist());
            elif('Hessian' in dir(config)):
                self.freq_min.append(np.sqrt(config.compute_vibrational_modes()).tolist());
            elif(calc_Hessian):
                config.compute_Hessian();
                self.freq_min.append(np.sqrt(config.compute_vibrational_modes()).tolist());
            else:
                self.freq_min.append([None]);
#        if(len(self.E_s) != 0):
#            self.reward.append(-int(self.fail[-1])-self.k_s*self.E_s[-1]-self.k_min*self.E_min[-1]-self.k_iter*self.iter_search[-1]);
 
    def add_reward(self, r):
        self.reward.append(r);
   
    def add_saddle(self, config, iteration=0, fail = False, calc_Hessian = False):
        
        if(fail):
            self.saddle.append(self.trajectory[-1]);
            self.E_s.append(self.E_min[-1]);
            self.freq_s.append(self.freq_min[-1]);
            
        else:
            self.saddle.append(config.atoms);
            self.E_s.append(config.potential());
            if('omega2' in dir(config)):
                omega2 = config.omega2;
            elif('Hessian' in dir(config)):
                omega2 = config.compute_vibrational_modes();
            elif(calc_Hessian):
                config.compute_Hessian();
                omega2 = config.compute_vibrational_modes();
            else:
                omega2 = [None];
            self.freq_s.append(list(omega2));
        
        self.fail.append(fail);
        self.iter_search.append(iteration);
        
    def add_action(self,sampling):
        self.action.append(sampling.action_details());
    
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
        
        