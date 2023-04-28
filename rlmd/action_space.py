import numpy as np;
from numpy.linalg import norm;
import ase;
from heapq import nsmallest;
from scipy.spatial import ConvexHull;

def actions(config, dist_mul_body = 1.5, dist_mul_surf = 1.2, collision_step = 5, collision_r = 0.5, act_mul = 1.5,act_mul_move = 1.2):
    ## TODO: please write a function to output a discrete set of actions
    ## actions is supposed to be a N*4 array, where N is the number of actions.
    ## The 1+3D array of each action: (i_atom, dx, dy, dz)
    Hlist = np.argwhere(config.atoms.get_atomic_numbers()==1);
    pos = config.atoms.get_positions();
    n_atom = len(pos);
    actions = [];

    Hlist = np.argwhere(config.atoms.get_atomic_numbers()==1);
    pos = config.atoms.get_positions();
    n_atom = len(pos);
    actions = [];
    
    for ind in Hlist:
        i = ind[0];
        otherHlist = Hlist.reshape(Hlist.shape[0])[np.nonzero(Hlist.reshape(Hlist.shape[0]) != i)];
        dist = config.atoms.get_distances(range(n_atom),i,mic=True);
        NN1  = nsmallest(4,dist+100*(config.atoms.get_atomic_numbers()==1))[3];
        c = pos[i];
        NN_ind = np.nonzero((dist!=0)*(dist<(NN1*dist_mul_body))*(config.atoms.get_atomic_numbers()!=1));
        NN = -config.atoms.get_distances(NN_ind,i,mic=True, vector = True);  
        NN_withH = np.append(np.array([[0,0,0]]),NN, axis=0);          
        hull = ConvexHull(NN_withH);
        surfaceOrnot = np.nonzero(hull.simplices==0)[0];
        NN = NN_withH;           
        if surfaceOrnot.size == 0:    #not on surface
            for simplex in hull.simplices:
                vec = np.mean(NN[simplex],axis=0)*1;
                flag0 = 1;                
                for move in range(collision_step+1):
                    atom_displaced = config.atoms.copy();
                    pos_displaced = atom_displaced.get_positions();
                    pos_displaced[i] += (1+(act_mul-1)*move/(collision_step))*vec;
                    atom_displaced.set_positions(pos_displaced);
                    pos1 = atom_displaced.get_distances(range(n_atom),i,mic=True);
                    
                    secondaryAtom_ind = np.nonzero(pos1 < collision_r)[0];
                    secondaryAtom_ind = secondaryAtom_ind[np.nonzero(1-np.isin(secondaryAtom_ind,NN_ind))];
                    secondaryAtom_ind = secondaryAtom_ind[np.nonzero(secondaryAtom_ind-i)];
                    secondaryAtom_hasH = secondaryAtom_ind[np.nonzero(np.isin(secondaryAtom_ind,otherHlist))];
                    if secondaryAtom_hasH.size != 0:
                        flag0 = 0;
                        break;
                    else:
                        if secondaryAtom_ind.size == 0:
                            continue;
                        else:
                            flag0 = 0;
                            NN_coll = -config.atoms.get_distances(secondaryAtom_ind,i,mic=True, vector = True);
                            NN_new = np.append(NN[simplex],NN_coll,axis = 0);
                            hull_local = ConvexHull(NN_new);
                            simpl = np.sort(hull_local.simplices,axis = 1);
                            for simplex_local in simpl:
                                if np.linalg.norm(simplex_local-np.array([0,1,2]))!=0:
                                    vec_act = np.mean(NN_new[simplex_local],axis=0)*1;
                                    all_dist = np.linalg.norm(pos - (c + act_mul*vec_act), axis =1);
                                    if(np.min(np.delete(all_dist,i))>collision_r):
                                        act = act_mul*vec_act;
                                        actions.append([i]+act.tolist());
                            break;
                if flag0:
                    act = act_mul*vec;
                    actions.append([i]+act.tolist()); 
                    
                    
        else:                         #there is a surface
            NN1  = nsmallest(3,dist+100*(config.atoms.get_atomic_numbers()==1))[2];
            NN_ind = np.nonzero((dist!=0)*(dist<(NN1*dist_mul_surf))*(config.atoms.get_atomic_numbers()!=1));
            NN = -config.atoms.get_distances(NN_ind,i,mic=True, vector = True);   
            NN_surface = np.append(np.array([[0,0,0]]),NN, axis=0);  
            hull_surface = ConvexHull(NN_surface); 
            simp_surface = np.sort(hull_surface.simplices,axis = 1);
            FromHOrnot = np.nonzero(simp_surface == 0)[0];
            NN = NN_surface;
            surface_ind = np.array([]);
            action_temp = [];
            #Find all the surface atoms and Deal with hopping into deep surface
            for spc in range(simp_surface.shape[0]):
                if np.isin(spc,FromHOrnot):
                    surface_ind = np.append(surface_ind, simp_surface[spc],axis = 0);   
                    pp = np.nonzero(np.sum((simp_surface == simp_surface[spc][1])+(simp_surface == simp_surface[spc][2]),axis = 1)==2)[0];
                    surf_tri_simp = simp_surface[pp[np.nonzero(pp != spc)]][0];
                    surf_tri_simp_atom = NN_surface[surf_tri_simp];
                    a = surf_tri_simp_atom[0]; b = surf_tri_simp_atom[1]; cc = surf_tri_simp_atom[2]; 
                    A = np.array([[np.dot(a-cc,a-cc),np.dot(a,b)-np.dot(b,cc)-np.dot(a,cc)+np.dot(cc,cc)],[np.dot(a,b)-np.dot(b,cc)-np.dot(a,cc)+np.dot(cc,cc),np.dot(b-cc,b-cc)]]);
                    B = np.array([np.dot(cc,cc)-np.dot(a,cc),np.dot(cc,cc)-np.dot(b,cc)])
                    coef_arr = np.linalg.solve(A,B);
                    
                    vec = -(coef_arr[0]*a+coef_arr[1]*b+(1-coef_arr[0]-coef_arr[1])*cc)+(NN_surface[simp_surface[spc][1]]+NN_surface[simp_surface[spc][2]])/2;
                    if(np.min(np.linalg.norm(pos - (c + act_mul*vec), axis =1))>collision_r):
                        act = act_mul*vec;
                        actions.append([i]+act.tolist());   
                else:
                    simplex = simp_surface[spc];
                    vec = np.mean(NN_surface[simplex],axis=0)*1;
                    flag1 = 1;
                    for move in range(collision_step):
                        atom_displaced = config.atoms.copy();
                        pos_displaced = atom_displaced.get_positions();
                        pos_displaced[i] += (1+(act_mul-1)*move/(collision_step))*vec;
                        atom_displaced.set_positions(pos_displaced);
                        pos1 = atom_displaced.get_distances(range(n_atom),i,mic=True);

                        secondaryAtom_ind = np.nonzero(pos1 < collision_r)[0];
                        secondaryAtom_ind = secondaryAtom_ind[np.nonzero(1-np.isin(secondaryAtom_ind,NN_ind))];
                        secondaryAtom_ind = secondaryAtom_ind[np.nonzero(secondaryAtom_ind-i)];
                        secondaryAtom_hasH = secondaryAtom_ind[np.nonzero(np.isin(secondaryAtom_ind,otherHlist))];
                        if secondaryAtom_hasH.size != 0:
                            flag1 = 0;
                            break;
                        else:
                            if secondaryAtom_ind.size == 0:
                                continue;
                            else:
                                flag1 = 0;
                                NN_coll = -config.atoms.get_distances(secondaryAtom_ind,i,mic=True, vector = True);
                                NN_new = np.append(NN_surface[simplex],NN_coll,axis = 0);
                                hull_local = ConvexHull(NN_new);
                                simpl = np.sort(hull_local.simplices,axis = 1);
                                for simplex_local in simpl:
                                    if np.linalg.norm(simplex_local-np.array([0,1,2]))!=0:
                                        vec_act = np.mean(NN_new[simplex_local],axis=0)*1;
                                        all_dist = np.linalg.norm(pos - (c + act_mul*vec_act), axis =1);
                                        if(np.min(np.delete(all_dist,i))>collision_r):
                                            act = act_mul*vec_act;
                                            action_temp.append([i]+act.tolist());
                                break;
                    if flag1:
                        act = act_mul*vec;
                        action_temp.append([i]+act.tolist());                
            surface_ind = np.sort(np.unique(surface_ind));
            surface_ind = surface_ind.astype(np.int64);
            
            
            if surface_ind.size == NN_surface.shape[0]:    # no inside atom is found

                vec = np.mean(NN_surface[1:NN_surface.shape[0]],axis=0)*1;

                act_mul_surf = 3/np.linalg.norm(vec);
                
                flag2 = 1;
                for move in range(collision_step+1):
                    atom_displaced = config.atoms.copy();
                    pos_displaced = atom_displaced.get_positions();
                    pos_displaced[i] += (1+(act_mul_surf-1)*move/(collision_step))*vec;
                    atom_displaced.set_positions(pos_displaced);
                    pos1 = atom_displaced.get_distances(range(n_atom),i,mic=True);

                    secondaryAtom_ind = np.nonzero(pos1 < collision_r)[0];
                    secondaryAtom_ind = secondaryAtom_ind[np.nonzero(1-np.isin(secondaryAtom_ind,NN_ind))];
                    secondaryAtom_ind = secondaryAtom_ind[np.nonzero(secondaryAtom_ind-i)];
                    secondaryAtom_hasH = secondaryAtom_ind[np.nonzero(np.isin(secondaryAtom_ind,otherHlist))];
                    if secondaryAtom_hasH.size != 0:
                        flag2 = 0;
                        break;
                    else:
                        if secondaryAtom_ind.size == 0:
                            continue;
                        else:
#                            print('collision: '+str(secondaryAtom_ind))
                            flag2 = 0;
                            NN_coll = -config.atoms.get_distances(secondaryAtom_ind,i,mic=True, vector = True);
                            NN_new = np.append(NN_surface,NN_coll,axis = 0);
                            hull_local = ConvexHull(NN_new);
                            uu = NN_surface[simplex].shape[0];
                            simpl = np.sort(hull_local.simplices,axis = 1);
                            for simplex_local in simpl:
                                if np.sum(simplex_local < surface_ind.size)<3:
                                    vec_act = np.mean(NN_new[simplex_local],axis=0)*1;
                                    vec_add = np.cross(NN_new[simplex_local][1]-NN_new[simplex_local][0],NN_new[simplex_local][2]-NN_new[simplex_local][0]);
                                    vec_add = vec_add*np.dot(vec_add,vec_act)/np.linalg.norm(vec_add)**2*(act_mul_move-1);
                                    vec_sum = vec_act + vec_add;
                                    all_dist = np.linalg.norm(pos - (c + vec_sum), axis =1);
                                    if(np.min(np.delete(all_dist,i))>collision_r):
                                        act = vec_sum;
                                        actions.append([i]+act.tolist());
                            break;
                if flag2:
                    act = act_mul*vec;
                    actions.append([i]+act.tolist()); 
            else:                                        # inside atom is found
                actions = actions + action_temp;
    return actions;

