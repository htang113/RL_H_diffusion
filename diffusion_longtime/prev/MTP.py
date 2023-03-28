# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 20:42:48 2022

@author: 17000
"""

import numpy as np;
import torch;
from torch import nn;

class moment_tensor_model(nn.Module):
    
    def __init__(self, parameters = [], seed = None, rmin = 1.5, rmax = 3.5, Nr = 4, mu_max = 3, nv_max = 3, Nb = 21):
        super().__init__();
        self.rmin = rmin;
        self.rmax = rmax;
        self.Nr = Nr;
        self.mu_max = mu_max;
        self.nv_max = nv_max;
        self.Nb = Nb;
        self.seed = seed;
        if(parameters == []):
            if(seed != None):
                torch.manual_seed(seed);
            self.parameters = [torch.tensor((torch.randn(mu_max, Nr+1,dtype=torch.float32)/np.sqrt(Nr)).tolist(),requires_grad=True),
                           torch.tensor((torch.randn(Nb,dtype=torch.float32)/np.sqrt(Nb)).tolist(),requires_grad=True)];
        else:
            self.parameters = parameters;
            
        self.ind = ['ij,klj->ikl','ij,kljm->iklm','ij,kljmn->iklmn',
                    'ij,kljmnu->iklmnu','ij,kljmnuv->iklmnuv'];
       
    def convert(self, traj):
        natm = len(traj[0].numbers);
        pos_list = [at.get_positions() for at in traj];
        lat = traj[0].cell.tolist();
        lat_vec = np.array([lat[0][0],lat[1][1],lat[2][2]]);
        self.in_data = [torch.zeros([len(pos_list),natm,self.Nr+1]+ [3]*i) for i in range(self.nv_max+1)];
        for p in range(len(pos_list)):
            pos = pos_list[p];
            for i in range(natm):
                for j in range(natm):
                    vec = np.array(pos[j])-np.array(pos[i]);
                    c1 = (vec)%lat_vec<self.rmax;
                    c2 = (lat_vec-vec)%lat_vec<self.rmax;
                    if((c1 + c2).all() and i != j):
                        vec = -(c2*(-vec + lat_vec)%lat_vec)+c1*vec%lat_vec;
                        r = np.linalg.norm(vec);
                        if(r<self.rmax):
                            Q = (self.rmax-r)**2*np.polynomial.chebyshev.chebvander((r-(self.rmax+self.rmin)/2)/(self.rmax-self.rmin)*2,self.Nr)[0];
                            r = torch.tensor(vec);
                            Q = torch.tensor(Q);
                            res = [Q, torch.outer(Q,r),
                                    torch.einsum('i,j,k->ijk',[Q,r,r]),
                                    torch.einsum('i,j,k,l->ijkl',[Q,r,r,r])];
#                                    torch.einsum('i,j,k,l,m->ijklm',[Q,r,r,r,r])];
                            for u in range(self.nv_max + 1):
                                self.in_data[u][p,i] += res[u];
        return self.in_data;
    
    def set_reference(self, config):
        self.reference = [dp[0,0] for dp in self.convert([config])];
        return self.reference;
        
    def forward(self, in_data):
        if('reference' in dir(self)):
            M=[torch.einsum(self.ind[nv], [self.parameters[0],in_data[nv]-self.reference[nv]]) for nv in range(self.nv_max+1)];
        else:
            M=[torch.einsum(self.ind[nv], [self.parameters[0],in_data[nv]]) for nv in range(self.nv_max+1)];
        
        B = torch.stack([M[1][0],
              M[1][1],
              M[1][2],
              torch.einsum('ij,ijk->ijk',[M[0][0],M[1][0]]),
              torch.einsum('ij,ijk->ijk',[M[0][0],M[1][1]]),
              torch.einsum('ij,ijk->ijk',[M[0][1],M[1][0]]),
              torch.einsum('ij,ijk->ijk',[M[0][0]**2,M[1][0]]),
              torch.einsum('ij,ijk->ijk',[M[0][0]**3,M[1][0]]),
              torch.einsum('ij,ijk->ijk',[M[0][0]**4,M[1][0]]),
              torch.einsum('ijk,ijkl->ijl',[M[1][0], M[2][0]]),
              torch.einsum('ijk,ijkl->ijl',[M[1][1], M[2][0]]),
              torch.einsum('ijk,ijkl->ijl',[M[1][0], M[2][1]]),
              torch.einsum('ij,ijk,ijkl->ijl',[M[0][0], M[1][0], M[2][0]]),
              torch.einsum('ij,ijk,ijkl->ijl',[M[0][0]**2, M[1][0], M[2][0]]),
              torch.einsum('ijk,ijk,ijl->ijl',[M[1][0], M[1][0], M[1][0]]),
              torch.einsum('ij,ijk,ijk,ijl->ijl',[M[0][0], M[1][0], M[1][0], M[1][0]]),
              torch.einsum('ijkl,ijklm->ijm',[M[2][0], M[3][0]]),
              torch.einsum('ij,ijkl,ijklm->ijm',[M[0][0],M[2][0], M[3][0]]),
              torch.einsum('ijkl,ijkl,ijm->ijm',[M[2][0], M[2][0], M[1][0]]),
              torch.einsum('ijkl,ijlm,ijm->ijk',[M[2][0], M[2][0], M[1][0]]),
              torch.einsum('ijklm,ijl,ijm->ijk',[M[3][0], M[1][0], M[1][0]])
              ],dim=0);   
        return torch.einsum('i,ijkl->jkl',self.parameters[1],B);
