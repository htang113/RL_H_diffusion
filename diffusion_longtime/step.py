import numpy as np;
from numpy.linalg import norm;
from scipy.optimize import minimize;
from time import time
from configuration import configuration;
import ase;
from ase.neb import NEB
from ase.optimize import MDMin,BFGS, FIRE
from ase.calculators.eam import EAM;

class saddle_search(object):
    
    def __init__(self, config, action: list=[], method: str='', threshold = 10**-2, max_iter = 10, step = 0.02):
        self.config = config
        self.action = action/norm(action);
        self.method = method
        self.threshold = threshold # eV/Ang
        self.max_iter = max_iter
        self.pos = self.config.pos('frac')
        self.cell_size = self.config.atoms.cell.tolist()[0][0];
        self.step = step/self.cell_size; #Ang
        self.n_atom = config.n_atom;
        
    def proj(self,force,e_min):
        alpha = np.matrix(np.ones(self.n_atom));
        e_o = e_min.reshape(self.n_atom,3);
        e_o -= alpha.T*(alpha*e_o/self.n_atom);
        e_o /= norm(e_o);
        f_proj = force-np.sum(force*e_o)*e_o;
        f_proj -= alpha.T*(alpha*f_proj/self.n_atom);
        return f_proj

    def get_step(self, force, method="optimal"):
        F = abs(np.sum(force.reshape(-1)*self.eigen['eigenvector']));
        print(F);
        #F is supposed to be projected onto e_min, and proj actually project it to its perpendicular hyperplane
        #norm(self.proj(force,self.eigen['eigenvector']))
        if method == "optimal":
            self.step = 2*F/(abs(self.eigen["eigenval"])*(1+np.sqrt(1+4*F**2/self.eigen["eigenval"]**2)))/self.cell_size;
        if self.step>0.3/self.cell_size:
            self.step=0.3/self.cell_size;
        return

    def relax(self,vec=[],step=None,init=False,method="CG",accuracy = 0.01):
        def f(x):
            x = x.reshape(self.config.n_atom, 3);
            self.config.set_atoms(x);
            return self.config.potential()

        def df(x):
            x = x.reshape(self.config.n_atom, 3);
            self.config.set_atoms(x);
            force = self.config.force();
            f_proj = self.proj(force,e_min);
            return -f_proj.reshape(-1)*self.cell_size;
        
        def df_init(x):
            x = x.reshape(self.config.n_atom, 3);
            self.config.set_atoms(x);
            force = self.config.force();
            return -force.reshape(-1)*self.cell_size;

        x0 = np.array(self.pos).reshape(-1)
        if init:
            if step is not None:
                x0 = np.array(self.pos).reshape(-1)+step
            res = minimize(f, x0, jac=df_init, method = method,options={'gtol':accuracy})
        else:
            e_min = vec;
            res = minimize(f, x0, jac=df, method = method,options={'gtol':accuracy})
        self.pos = res.x.reshape(self.config.n_atom,3)
        self.config.set_atoms(self.pos);
        return res.x
        
    def search(self):
        fail = True;
        i = 0;
        j_record = -1;
        accuracy = 0.05;
        # self.traj = [];
        self.relax(init=True,method="CG",accuracy=accuracy);
        # self.traj.append(self.config.atoms);
        while i <= self.max_iter:
            if(j_record<0 or j_record%3 ==0):
                self.eigen = self.config.get_smallest_eigen(method = 'BFGS', accuracy = 10**-2);
    
            if self.eigen["imag"]:
                vec = self.eigen["eigenvector"];
            else:
                vec = self.action;
            force = self.config.force();
            
            if self.eigen["imag"]:
                print("eigenvalue is neg")
                residue = norm(force,ord=np.inf);
                print('force deviation = '+str(residue));
                if residue<self.threshold:
                    fail = False
                    break
                accuracy = max(0.2*abs(np.dot(force.reshape(-1),vec)),self.threshold/3);
                self.get_step(force,method="optimal");
                action  = -np.dot(force.reshape(-1),vec)*vec;
                action = (action/norm(action)).reshape(self.config.n_atom,3);
                self.pos = self.pos + action * self.step;
                self.action_real = action;
                j_record += 1;
            else:
                print("eigenvalue is pos")
                accuracy = 0.05;
                self.pos = self.pos + self.action * self.step;
            
            print('relaxing')
            self.relax(vec,init=False,method="CG",accuracy=accuracy);
            print('learning rate = '+str(self.step))
            self.config.set_atoms(self.pos);

            # self.traj.append(self.config.atoms);
            i += 1;
            
        f_eff = self.config.F_eff(action);
        return {'Atom_Pos':self.pos, 'F_eff':f_eff, 'Fail':fail, 'Iterations':i}
    
    
class move(object):
    
    def __init__(self, config, action: list=[], method: str='', threshold = 10**-2, max_iter = 10, step = 0.02):
        self.config = config
        self.action = action/norm(action);
        self.method = method
        self.threshold = threshold # eV/Ang
        self.max_iter = max_iter
        self.pos = self.config.pos('cartesion').tolist();
        self.cell_size = self.config.atoms.cell.tolist()[0][0];
        self.step0 = step; #Ang
        self.n_atom = config.n_atom;
        
    def relax(self,vec=[],step0=None,init=False,accuracy = 0.1):
        if init:
            dyn = MDMin(self.config.atoms);
            dyn.run(fmax=accuracy);
        else:
   #         c = ase.constraints.FixCom();
            f = ase.constraints.FixAtoms(indices=[self.act_atom]);
   #         self.config.atoms.set_constraint(c);
            self.config.atoms.set_constraint(f);
            dyn = MDMin(self.config.atoms);
            dyn.run(fmax=accuracy);
            del self.config.atoms.constraints; 
        self.pos = self.config.atoms.get_positions().tolist();
        return self.pos;
    
    def mask(self):
        axis = self.config.atoms[self.act_atom].position;
        rl = [atom.position-axis for atom in self.config.atoms]
        rl_abs = [[min(x,self.cell_size-x) for x in r] for r in rl];
        mask = [np.linalg.norm(r)<4 for r in rl_abs];
        return mask;
    
    def step(self, accuracy = 0.05):
        
        self.relax(init=True,accuracy=accuracy);
        
        self.act_atom = np.argmax(np.linalg.norm(self.action,axis=1));
        self.pos_last, energy0 = self.config.pos(), self.config.potential();
        self.initial = self.config.atoms.copy();
                
        E0,E1  = 0, energy0;
        fail = True;
        for n_step in range(6):
            self.pos = (self.config.pos(f='cartesion') + self.step0*self.action).tolist();
            self.config.set_atoms(self.pos,convention = 'cartesion');
            self.config.atoms.set_constraint(ase.constraints.FixAtoms(mask=self.mask()))
            self.relax(init=False,accuracy = 5*accuracy);
            E0, E1 = E1, self.config.potential();
            if(E1<E0):
                fail = False;
                break;               
            if(E1-energy0>5):
                break;
            print(E1);
            
        if(not fail):        
            self.relax(init=True,accuracy = accuracy);
            self.config.plot_configuration();
            if(norm(self.pos_last - self.config.pos())<0.2):
                fail=True;
            pos_out = self.config.atoms.get_scaled_positions();
        else:
            pos_out = self.pos_last;
            self.config.set_atoms(self.pos_last);
            
        return {'Atom_Pos':pos_out, 'Fail':fail}
        
    def saddle(self, n_points=10):
        
        images = [self.initial];
        images += [self.initial.copy() for i in range(n_points-2)];
        images += [self.config.atoms];
        
        neb = NEB(images)
        neb.interpolate()
        for image in range(n_points):
            images[image].calc = EAM(potential='Cu01.eam.alloy');
            
        optimizer = MDMin(neb)
        res = optimizer.run(fmax=0.3,steps = 40)
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
                pos_out = images[i].get_positions();
                
            output = configuration(self.config.box,pos_out/self.cell_size);
            output.set_potential(platform ='ase', potential_id = 'Cu01.eam.alloy')
        else:
            output = configuration(self.config.box,images[0].get_positions()/self.cell_size);
            output.set_potential(platform ='ase', potential_id = 'Cu01.eam.alloy')
        return {'saddle':output , 'images':images, 'Fail':not res};
    
    
