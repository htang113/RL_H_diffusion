import numpy as np;
from numpy.linalg import norm;
from configuration import configuration;
import ase;
from heapq import nsmallest;
from scipy.spatial import ConvexHull;

def actions(config):
    ## TODO: please write a function to output a discrete set of actions
    ## actions is supposed to be a N*4 array, where N is the number of actions.
    ## The 1+3D array of each action: (i_atom, dx, dy, dz)
    Hlist = np.argwhere(config.atoms.get_atomic_numbers()==1);
    pos = config.atoms.get_positions();
    n_atom = len(pos);
    actions = [];

    for ind in Hlist:
        i = ind[0];
        dist = config.atoms.get_distances(range(n_atom),i);
        NN1  = nsmallest(2,dist)[1];
        c = pos[i];
        NN = [q[0] for q in np.argwhere((dist!=0)*(dist<(NN1*1.4)))];
        NN = pos[NN]-c;
        hull = ConvexHull(NN);
        for simplex in hull.simplices:
            vec = np.mean(NN[simplex],axis=0)*1.4;
            actions.append([i]+vec.tolist());
        
    return actions;

