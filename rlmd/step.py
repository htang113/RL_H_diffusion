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

class environment(object):
    
    def __init__(self, config, max_iter = 50,cutoff = 4):
        
        self.config = config
        self.max_iter = max_iter
        self.pos = self.config.pos('cartesion').tolist();
        self.cell = self.config.atoms.cell.tolist();
        self.n_atom = len(self.pos);
        self.cutoff = cutoff;
        self.output = StringIO();
        
    def relax(self, accuracy = 0.05):
        
        self.config.atoms.set_constraint(ase.constraints.FixAtoms(mask = [False]*self.n_atom))
        dyn = MDMin(self.config.atoms, logfile='log');
        with redirect_stdout(self.output):
            dyn.run(fmax=accuracy,steps = self.max_iter);
        self.pos = self.config.atoms.get_positions().tolist();
        
        return self.pos;
    
    def step(self, action: list=[], accuracy = 0.05):
        
        self.action = action;
        self.pos_last, energy0 = self.config.pos(f='cartesion'), self.config.potential();
        self.initial = self.config.atoms.copy();
        
        self.act_atom = self.action[0];
        self.act_displace = np.array([[0,0,0]]*self.act_atom+[self.action[1:]]+[[0,0,0]]*(self.n_atom-1-self.act_atom));
        self.pos = (self.config.pos(f='cartesion') + self.act_displace).tolist();
        self.config.set_atoms(self.pos,convention = 'cartesion');
        
        self.relax(accuracy = accuracy);
        fail = int(norm(self.pos_last - self.config.pos(f='cartesion'))<0.2);
        E_next = self.config.potential();
            
        return E_next, fail;

    def mask(self):
        
        dist = self.config.atoms.get_distances(range(self.n_atom),self.act_atom);
        mask = [d>self.cutoff for d in dist];
        
        return mask;
    
    def saddle(self, accuracy = 0.1, n_points=10):
        
        self.config.atoms.set_constraint(ase.constraints.FixAtoms(mask = [False]*self.n_atom));
        self.initial.set_constraint(ase.constraints.FixAtoms(mask = [False]*self.n_atom));
        images = [self.initial];
        images += [self.initial.copy() for i in range(n_points-2)];
        images += [self.config.atoms];

        neb = NEB(images)
        neb.interpolate()
        
        temp = configuration([1]*6,[(0,0,0)],'H');
        for image in range(n_points):
            temp.set_potential();
            images[image].calc = temp.calculator;
            images[image].set_constraint(ase.constraints.FixAtoms(mask=self.mask()));
        with redirect_stdout(self.output):  
            optimizer = MDMin(neb)
        
        res = optimizer.run(fmax=accuracy,steps = self.max_iter);
        if(res):
            Elist = [image.get_potential_energy() for image in images];
            E = np.max(Elist);
        else:
            E = 0;
            
        self.config.atoms.set_constraint(ase.constraints.FixAtoms(mask = [False]*self.n_atom));
        
#        pos_frac = self.config.atoms.get_scaled_positions()%1;
#        self.config.atoms.set_scaled_positions(pos_frac);
        
        return E, int(not res);