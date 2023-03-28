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
    def __init__(self, box: list, pos: list, element: str, supercell: tuple = (1,1,1), pbc: list = (True,True,True)): 
        # set computation supercell box = [a, b, c, alpha, beta, gamma]. unit: A, degree
        self.atoms = ase.Atoms(element,
              cell=box,
              pbc=pbc,
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
        element = self.atoms.get_atomic_numbers();
        cell    = self.atoms.cell;
        pbc     = self.atoms.pbc;
        if(slab):
            cell[2,2] *= 1 + 10/norm(cell[2,2]);
        if(convention == 'frac'):
            self.atoms = ase.Atoms(element,
              cell=cell,
              pbc = pbc,
              scaled_positions=pos);
        else:
            self.atoms = ase.Atoms(element,
              cell=cell,
              pbc = pbc,
              positions=pos);
        self.atoms.calc = self.calculator;
        return self.atoms;
    
    def remove_atom(self, atom_index):
        element = self.atoms.get_atomic_numbers().tolist();
        cell    = self.atoms.cell;
        pbc     = self.atoms.pbc;
        pos = self.atoms.get_scaled_positions().tolist();
        del pos[atom_index];
        del element[atom_index];
        self.atoms = ase.Atoms(element,
              cell=cell,
              pbc = pbc,
              scaled_positions=pos);
        self.atoms.calc = self.calculator;
        return self.atoms;
    
    def add_atom(self, frac_coords, atomic_number):
        pos = self.atoms.get_scaled_positions().tolist();
        pos.append(frac_coords);
        element = self.atoms.get_atomic_numbers().tolist();
        element.append(atomic_number);
        cell    = self.atoms.cell;
        pbc     = self.atoms.pbc;
        self.atoms = ase.Atoms(element,
              cell=cell,
              pbc = pbc,
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
