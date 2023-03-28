import numpy as np;
from numpy.linalg import norm;
from scipy.optimize import minimize;
from time import time
from configuration import configuration;
import ase;
from ase.neb import NEB
from ase.optimize import MDMin,BFGS, FIRE
from ase.calculators.eam import EAM;

def action_space(config):
    ## TODO: please write a function to output a discrete set of actions
    ## actions is supposed to be a N*4 array, where N is the number of actions.
    ## The 1+3D array of each action: (i_atom, dx, dy, dz)
    
    return actions;

