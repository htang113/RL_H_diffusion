# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:32:08 2022

@author: 17000
"""

import ase;
import numpy as np;
import scipy;
from itertools import chain;
from scipy.sparse import linalg;

class configuration(object):
    def __init__(self, box: list, pos: list, supercell: tuple = (1,1,1), element: str = 'Cu', n_atom: int = 4, pbc: list = (True,True,True)): 
        # set computation supercell box = [a, b, c, alpha, beta, gamma]. unit: A, degree
        self.box = [box[i]*supercell[i] for i in range(3)]+box[3:];
        self.element = element;
        self.n_atom = len(pos)*supercell[0]*supercell[1]*supercell[2];
        self.pbc = pbc;
        
        self.atoms = ase.Atoms(self.element+str(len(pos)),
              cell=box,
              pbc=self.pbc,
              scaled_positions=pos)*supercell;
            
    def set_potential(self, platform: str = 'matlantis', potential_id: str = 'EAM_Dynamo_MendelevKing_2013_Cu__MO_748636486270_005'):
        self.platform = platform;
        if(platform == 'matlantis'):
            from pfp_api_client.pfp.estimator import Estimator
            from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
            estimator = Estimator(model_version="v2.0.0");
            self.calculator = ASECalculator(estimator);
            
        elif(platform == 'kimpy'):
            from ase.calculators.kim.kim import KIM;
            self.calculator = KIM(potential_id);
        
        elif(platform =='ase'):
            from ase.calculators.eam import EAM;
            self.calculator = EAM(potential=potential_id);
        else:
            raise('Error: platform should be set as either matlantis or kimpy');
        self.atoms.calc = self.calculator;
        
    def set_atoms(self, pos:list, convention = 'frac', slab = False):    # set atomic positions in the configuration by fraction coordinate list pos = [\vec x_1, ..., \vec x_N]
        if(slab):
            self.box[2] *= 1 + 10/norm(self.box[2]);
        if(convention == 'frac'):
            self.atoms = ase.Atoms(self.element+str(self.n_atom),
              cell=self.box,
              pbc=self.pbc,
              scaled_positions=pos);
        else:
            self.atoms = ase.Atoms(self.element+str(self.n_atom),
              cell=self.box,
              pbc=self.pbc,
              positions=pos);
        self.atoms.calc = self.calculator;
        return self.atoms;
    
    def remove_atom(self, atom_index):
        pos = self.atoms.get_scaled_positions().tolist();
        del pos[atom_index];
        self.n_atom -= 1;
        self.atoms = ase.Atoms(self.element+str(self.n_atom),
              cell=self.box,
              pbc=self.pbc,
              scaled_positions=pos);
        self.atoms.calc = self.calculator;
        return self.atoms;
    
    def add_atom(self, frac_coords):
        pos = self.atoms.get_scaled_positions().tolist();
        pos.append(frac_coords);
        self.n_atom += 1;
        self.atoms = ase.Atoms(self.element+str(self.n_atom),
              cell=self.box,
              pbc=self.pbc,
              scaled_positions=pos);
        self.atoms.calc = self.calculator;
        return self.atoms;
    
    def pos(self, f = 'frac'):
        if(f[0] == 'f'):
            return self.atoms.get_scaled_positions();
        else:
            return self.atoms.get_positions();
        
    def force(self):     # compute force on each atom f = [\vec f_1, ..., \vec f_N]. unit: eV/A
        return self.atoms.get_forces();
    
    def potential(self):   # compute total potential energy. unit: eV
        return self.atoms.get_potential_energy();
    
    def plot_configuration(self):  # plot the atomic configuration
        if(self.platform =='matlantis'):
            import nglview as nv;
            v = nv.show_ase(self.atoms, viewer="ngl");
            v.add_label(
                color="blue", labelType="text",
                labelText=[self.atoms[i].symbol + str(i) for i in range(self.atoms.get_global_number_of_atoms())],
                zOffset=1.0, attachment='middle_center', radius=1.0
            );
        else:
            from ase.visualize import view;
            c = self.atoms;
            v = view(c);
        return v;
    
    def compute_Hessian(self, step: float = 0.01):
        # we construct a projection operator to transform to mass center coordinate frame
        # you do not need to understand the details
        if('projector' not in dir(self)):
            one_direction = [[1]*i+[-i]+[0]*(self.n_atom-i-1) for i in range(1, self.n_atom)];
            self.projector = np.matrix([[one_direction[i//3][j//3]*(i%3 == j%3)/np.sqrt((i//3+2)*(i//3+1)) for j in range(3*self.n_atom)] for i in range(3*self.n_atom-3)])

        pos = self.atoms.get_scaled_positions();
        f0 = self.force();
        self.Hessian = [];
        for atom in range(len(pos)):
            for i in range(3):
                b_matrix = self.atoms.cell.reciprocal().tolist();
                for j in range(3):
                    pos[atom][j] += step*b_matrix[j][i]; 
                self.set_atoms(pos);
                self.Hessian.append(list(chain(*-(self.force()-f0)/step)));
                for j in range(3):
                    pos[atom][j] -= step*b_matrix[j][i];
        self.set_atoms(pos);
        return self.Hessian;
    
    def get_smallest_eigen(self, method = 'CG', step = 0.05, accuracy = 0.001, krylov_dim = 20):
        if('Hessian' in dir(self)):
            H_projected = scipy.sparse.coo_matrix(self.projector*self.Hessian*self.projector.T);
            res = linalg.eigsh(H_projected, k =1, which = 'SA');
            return {'eigenvalue':res[0],'eigenvector':self.projector.T*res[1]};
        elif(method == 'krylov'):
            alpha = np.matrix(np.ones(self.n_atom));
            if('eigen_min' in dir(self)):
                v = self.eigen_min;
            else:
                v = np.random.randn(self.n_atom,3);
                v -= alpha.T*(alpha*v/self.n_atom);
                v = v/np.linalg.norm(v);
            b_matrix = np.matrix(self.atoms.cell.reciprocal().tolist());
            krylov = [v];
            Hv_record = [];
            pos = np.array(self.atoms.get_scaled_positions());
            f0 = self.force();
            for iteration in range(krylov_dim-1):
                self.set_atoms(pos+krylov[-1]*b_matrix*step);
                Hv = -(self.force()-f0)/step;
                Hv_record.append(Hv[:,:]);
                v_next = np.array(Hv[:,:]);
                for v1 in krylov:
                    v_next -= np.sum(v_next*v1)*v1;
                v_next -= alpha.T*(alpha*v/self.n_atom);
                krylov.append(v_next/np.linalg.norm(v_next));
            self.set_atoms(pos+krylov[-1]*step);
            Hv = -(self.force()-f0)/step;
            Hv_record.append(Hv);
            H_sub = [[round(np.sum(krylov[i]*Hv_record[j]),2) for i in range(krylov_dim)] for j in range(krylov_dim)];
            H_sub = (np.matrix(H_sub)+np.matrix(H_sub).T)/2;
            self.H = H_sub;
            self.k = krylov;
            self.Hv = Hv_record;
            res = linalg.eigsh(scipy.sparse.coo_matrix(H_sub), k = 1, which ='SA');
            vec_lambda = np.sum((res[1].T)[0]*np.array(krylov).T,axis=2);
            self.eigen_min = vec_lambda;
            return {'imag':res[0][0]<0,'eigenvalue':64.654*np.sqrt(abs(res[0][0])),'eigenvector': vec_lambda.T};
# =============================================================================
#         elif(method == 'power'):
#             if('eigen_min' in dir(self)):
#                 v = self.eigen_min.reshape(self.n_atom,3);
#                 v -= alpha.T*(alpha*v/self.n_atom);
#                 v = v/np.linalg.norm(v);
#             else:
#                 v = np.random.randn(self.n_atom,3);
#                 v -= alpha.T*(alpha*v/self.n_atom);
#                 v = v/np.linalg.norm(v);
#             f0 = self.force();
#             pos = np.array(self.atoms.get_scaled_positions());
#             b_matrix = np.matrix(self.atoms.cell.reciprocal().tolist());
#             for i in range(20):
#                 self.set_atoms(pos+v*b_matrix*step);
#                 self.Hv = (self.force()-f0)/step;
#                 
# =============================================================================
        else:
            from scipy import optimize;
            alpha = np.matrix(np.ones(self.n_atom));
            if('eigen_min' in dir(self)):
                v = self.eigen_min.reshape(self.n_atom,3);
                v -= alpha.T*(alpha*v/self.n_atom);
                v = v/np.linalg.norm(v);
            else:
                v = np.random.randn(self.n_atom,3);
                v -= alpha.T*(alpha*v/self.n_atom);
                v = v/np.linalg.norm(v);
            f0 = self.force();
            pos = np.array(self.atoms.get_scaled_positions());
            b_matrix = np.matrix(self.atoms.cell.reciprocal().tolist());
            def f(v0):
                v0 = v0.reshape(self.n_atom,3);
                self.set_atoms(pos+v0*b_matrix*step);
                self.Hv = -(self.force()-f0)/step;
                self.Hv -= alpha.T*(alpha*self.Hv/self.n_atom);
                v_abs = np.linalg.norm(v0)
                return np.sum(self.Hv*v0)/2/v_abs**2;
            
            def df(v0):
                v0 = v0.reshape(self.n_atom,3);
                v_abs = np.linalg.norm(v0)
                return self.Hv.reshape(-1)/v_abs**2-np.sum(self.Hv*v0)/v_abs**4*v0.reshape(-1);
            
            res = optimize.minimize(f,v,jac=df,method=method,options={'gtol': accuracy, 'maxiter':200})
            eigenval = np.sum(self.Hv.reshape(-1)*res.x)/np.linalg.norm(res.x)**2;
            self.eigen_min = res.x/np.linalg.norm(res.x);
            self.set_atoms(pos);
            self.eig_res = {'imag':eigenval<0,'eigenval':eigenval,'eigenvector': self.eigen_min};
            return self.eig_res;
        
    def F_eff(self, action):
        f = self.force();
        if('eig_res' not in dir(self)):
            raise('Error: smallest eigen value has not been computed')
        elif(self.eig_res['imag'] == False):
            return f-2*np.sum(action*f)*action/np.linalg.norm(action)**2;
        else:
            lambda0 = self.eig_res['eigenvector'].reshape(self.n_atom,3)
            return f-2*np.sum(lambda0*f)*lambda0;
        
    def compute_vibration_modes(self):
        H_projected = self.projector*self.Hessian*self.projector.T;
        M_reduced = self.projector*np.diag([self.atoms.get_masses()[i//3] for i in range(self.n_atom*3)])*self.projector.T;
        omega2 = scipy.linalg.eigvalsh(H_projected, b =M_reduced);
        self.omega2 = 64.654**2*omega2;
        return self.omega2;
    
