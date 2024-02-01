import numpy as np;
from numpy.linalg import norm;
from scipy.optimize import minimize;
from time import time
from rlmd.configuration import configuration;
import ase;
from ase.neb import NEB
from ase.optimize import MDMin,BFGS, FIRE
from ase.calculators.eam import EAM;
from contextlib import redirect_stdout
from io import StringIO
from pfp_api_client.pfp.estimator import Estimator
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator

class environment(object):
    
    def __init__(self, config, logfile = 'log', max_iter = 100,cutoff = 4):
        
        self.config = config
        self.max_iter = max_iter
        self.pos = self.config.pos('cartesion').tolist();
        self.cell = self.config.atoms.cell.tolist();
        self.n_atom = len(self.pos);
        self.cutoff = cutoff;
        self.output = StringIO();
        self.logfile = logfile;
        
    def relax(self, accuracy = 0.05):
        
        self.config.atoms.set_constraint(ase.constraints.FixAtoms(mask = [False]*self.n_atom))
        dyn = MDMin(self.config.atoms, logfile=self.logfile);
        with redirect_stdout(self.output):
            converge = dyn.run(fmax=accuracy,steps = self.max_iter);
        self.pos = self.config.atoms.get_positions().tolist();
        
        return self.pos, converge;
    
    def step(self, action: list=[], accuracy = 0.05):
        
        self.action = action;
        self.pos_last, energy0 = self.config.pos(f='cartesion'), self.config.potential();
        self.initial = self.config.atoms.copy();
        
        self.act_atom = self.action[0];
        self.act_displace = np.array([[0,0,0]]*self.act_atom+[self.action[1:]]+[[0,0,0]]*(self.n_atom-1-self.act_atom));
        self.pos = (self.config.pos(f='cartesion') + self.act_displace).tolist();
        self.config.set_atoms(self.pos,convention = 'cartesion');
        
        _, converge = self.relax(accuracy = accuracy);
        fail = int(norm(self.pos_last - self.config.pos(f='cartesion'))<0.2) + (not converge);
        E_next = self.config.potential();
            
        return E_next, fail;
    
    def revert(self):
        self.config.set_atoms(self.pos_last,convention = 'cartesion');
    
    def mask(self):
        
        dist = self.config.atoms.get_distances(range(self.n_atom),self.act_atom);
        mask = [d>self.cutoff for d in dist];
        
        return mask;
    
    def normalize(self):
        pos_frac = self.config.atoms.get_scaled_positions()%1;
        self.config.atoms.set_scaled_positions(pos_frac);
        return 0;
    
    def saddle(self, moved_atom=-1, accuracy = 0.1, n_points=10, r_cut = 4):
        
        self.config.atoms.set_constraint(ase.constraints.FixAtoms(mask = [False]*self.n_atom));
        self.initial.set_constraint(ase.constraints.FixAtoms(mask = [False]*self.n_atom));
        images = [self.initial];
        images += [self.initial.copy() for i in range(n_points-2)];
        images += [self.config.atoms];

        neb = NEB(images)
        neb.interpolate()
        
        for image in range(n_points):
            images[image].calc = ASECalculator(Estimator(model_version="v4.0.0"));
            images[image].set_constraint(ase.constraints.FixAtoms(mask=self.mask()));
        with redirect_stdout(self.output):  
            optimizer = MDMin(neb, logfile=self.logfile)
        
        res = optimizer.run(fmax=accuracy,steps = self.max_iter);
        res = True;
        if(res):
            Elist = [image.get_potential_energy() for image in images];
            E = np.max(Elist);
            
            def log_niu_prod(input_atoms, NN, saddle=False):
                delta = 0.05;

                mass = input_atoms.get_masses()[NN];
                mass = np.array([mass[i//3] for i in range(3*len(NN))]);
                mass_mat = np.sqrt(mass[:,None]*mass[None,:]);
                Hessian = np.zeros([len(NN)*3,len(NN)*3]);
                f0 = input_atoms.get_forces();
                pos_init = input_atoms.get_positions();
                for u in range(len(NN)):
                    for j in range(3):
                        pos1 = pos_init.copy();
                        pos1[NN[u]][j] += delta;
                        input_atoms.set_positions(pos1);
                        Hessian[3*u+j] = -(input_atoms.get_forces()-f0)[NN].reshape(-1)/delta;
                
                freq_mat = (Hessian+Hessian.T)/2/mass_mat;
                if(saddle):
                    prod = np.prod(np.linalg.eigvals(freq_mat)[1:])
                else:
                    prod = np.linalg.det(freq_mat);
                output = np.log(np.abs(prod))/2;
                
                return output;
            
            max_ind = np.argmax(Elist);
            r_cut = 3;
            NN = self.initial.get_distances(moved_atom, range(len(self.initial)), mic=True);
            NN = np.argwhere(NN<r_cut).T[0];
            self.initial.calc =  ASECalculator(Estimator(model_version="v4.0.0"));
            self.initial.set_constraint(ase.constraints.FixAtoms(mask = [False]*self.n_atom));
            images[max_ind].calc = ASECalculator(Estimator(model_version="v4.0.0"));
            images[max_ind].set_constraint(ase.constraints.FixAtoms(mask = [False]*self.n_atom));
            log_niu_min = log_niu_prod(self.initial,NN);
            log_niu_s = log_niu_prod(images[max_ind],NN, saddle=True);
            
            log_attempt_freq = log_niu_min-log_niu_s+np.log(1.55716*10);

        else:
            E = 0;
            log_attempt_freq = 0;
        self.config.atoms.set_constraint(ase.constraints.FixAtoms(mask = [False]*self.n_atom));
        
        pos_frac = self.config.atoms.get_scaled_positions()%1;
        self.config.atoms.set_scaled_positions(pos_frac);
        
        return E, log_attempt_freq, int(not res);