import numpy as np;
from numpy.linalg import norm;
from scipy.optimize import minimize;
from time import time
from configuration import configuration;
import ase;
from ase.neb import NEB
from ase.optimize import MDMin,BFGS, FIRE
from ase.calculators.eam import EAM;

class environment(object):
    
    def __init__(self, config, action: list=[], max_iter = 50,cutoff = 4):
        self.config = config
        self.action = action;
        self.max_iter = max_iter
        self.pos = self.config.pos('cartesion').tolist();
        self.cell = self.config.atoms.cell.tolist();
        self.n_atom = len(self.pos);
        self.cutoff = cutoff;
        
    def relax(self, accuracy = 0.1):

        dyn = MDMin(self.config.atoms);
        dyn.run(fmax=accuracy);

        self.pos = self.config.atoms.get_positions().tolist();
        return self.pos;
    
    def step(self, accuracy = 0.05):
    
        self.pos_last, energy0 = self.config.pos(f='cartesion'), self.config.potential();
        self.initial = self.config.atoms.copy();
        
        self.act_atom = self.action[0];
        self.act_displace = np.array([[0,0,0]]*self.act_atom+[self.action[1:]]+[[0,0,0]]*(self.n_atom-1-self.act_atom));
        self.pos = (self.config.pos(f='cartesion') + self.act_displace).tolist();
        self.config.set_atoms(self.pos,convention = 'cartesion');
        
        self.relax(accuracy = accuracy);
        self.config.plot_configuration();
        fail =(norm(self.pos_last - self.config.pos(f='cartesion'))<0.2);
        pos_out = self.config.atoms.get_scaled_positions();
            
        return {'Atom_Pos':pos_out, 'Fail':fail}

    def mask(self):
        dist = self.config.atoms.get_distances(range(self.n_atom),self.act_atom);
        mask = [d>self.cutoff for d in dist];
        return mask;
    
    def saddle(self, accuracy = 0.1, n_points=10):
        
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
            
        optimizer = MDMin(neb)
        
        res = optimizer.run(fmax=accuracy,steps = self.max_iter);
        if(res):
            E = [image.get_potential_energy() for image in images];
            i = np.argmax(E);
            if(i>0 and i<n_points-1):
                A, B, C = (E[i+1]+E[i-1]-2*E[i])/2, (E[i+1]-E[i-1])/2, E[i];
                x0 = -B/2/A;
                if(x0>0):
                    pos_out = images[i].get_positions()*(1-x0)+images[i+1].get_positions()*x0;
                else:
                    pos_out = images[i].get_positions()*(1+x0)-images[i-1].get_positions()*x0;
            else:
                A, B, C = 1, 0, E[i];
                pos_out = images[i].get_scaled_positions();
                
            output = configuration(self.config.atoms.cell,pos_out,self.config.atoms.get_atomic_numbers());
            output.set_potential()
        else:
            output = configuration(self.config.atoms.cell,images[0].get_scaled_positions(),self.config.atoms.get_atomic_numbers());
            output.set_potential()
        
        self.config.atoms.set_constraint(ase.constraints.FixAtoms(mask = [False]*self.n_atom));
            
        return {'saddle':output , 'images':images, 'Fail':not res};
