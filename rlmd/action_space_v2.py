import numpy as np;
from numpy.linalg import norm;
import ase;
from heapq import nsmallest;
from scipy.spatial import ConvexHull;
from itertools import product;

def actions(config):
    
    a = 3.528;
    Hlist = np.argwhere(config.atoms.get_atomic_numbers()==1).T[0];
    metallist = np.argwhere(config.atoms.get_atomic_numbers()!=1).T[0];
    pos = config.atoms.get_positions();
    L = config.atoms.cell[0,0];
    n_atom = len(pos);
    actions = [];
    
    dist_mat = config.atoms.get_all_distances(mic=True);

    crit = np.sum(dist_mat[metallist][:,metallist]<2.495*1.2, axis=1);
    vacancy_l = np.argwhere(crit == 12).T[0];
    
    def distance(x1, yn):
        
        dr = (yn - x1+L/2)%L - L/2;
        
        return dr;
    
    def test(i, vec):
        
        test = config.atoms.copy();
        pos_test = test.get_positions();
        pos_test[i] += vec;
        test.set_positions(pos_test);
        
        return np.sum(test.get_distances(i,Hlist) < 0.7) == 1;
    
    mean = np.zeros(3);
    for i in vacancy_l:
        mean += distance(pos[vacancy_l[0]], pos[i]);
        
    pos_v = pos[vacancy_l[0]] + mean/len(vacancy_l);
    
    
    for i in vacancy_l:
        
        actions.append([i]+distance(pos[i],pos_v).tolist());
    
    for i in Hlist:
        
        if(np.linalg.norm(distance(pos[i],pos_v))>0.75*a):
            
            dist = config.atoms.get_distances(i, metallist, mic=True);
            NN = np.argwhere(dist < a/np.sqrt(2)).T[0];
            points = config.atoms.get_distances(i, NN, mic=True, vector=True);
            hull = ConvexHull(points);
            
            for simplex in hull.simplices:
                
                vec = np.mean(points[simplex],axis=0);
                vec = a*np.sqrt(3)/4*vec/np.linalg.norm(vec);
                
                vacant = test(i, vec);
                if(vacant):
                    actions.append([i]+vec.tolist());
            
        else:
            
            r_vi = distance(pos_v, pos[i]);
            
            dist = np.linalg.norm(r_vi);
            direction = np.argmax(np.abs(r_vi));
            
            for u in range(3):
                if(u!= direction):
                    vec = np.array([dist*(k==u) for k in range(3)])-r_vi;
                    if(test(i, vec)):
                        actions.append([i]+vec.tolist());
                    vec = np.array([-dist*(k==u) for k in range(3)])-r_vi;
                    if(test(i,vec)):
                        actions.append([i]+vec.tolist());
            
            for u,v in product([-1,1],[-1,1]):
                vec = distance(pos_v, pos[i]);
                vec = vec*a/4/np.linalg.norm(vec)+np.roll(np.array([0,u*a/4,v*a/4]), direction);
                if(test(i,vec)):
                    actions.append([i]+vec.tolist());
        
    return actions;

