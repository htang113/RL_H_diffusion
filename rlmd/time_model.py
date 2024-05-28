#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:08:57 2024

@author: ubuntu
"""

import numpy as np;
from numpy.linalg import norm;
import scipy;
import torch
from itertools import chain;
from itertools import product;
from scipy.sparse import linalg;
from torch import nn;
        
class T_NN(torch.nn.Module):
    
    def __init__(self, device, elements=[1,29],r_cut = 3.5, N_emb=32, N_fit = 64, atom_max = 30):
        super(T_NN, self).__init__()
        
        self.name = 'descriptorNN';
        # set computation supercell box = [a, b, c, alpha, beta, gamma]. unit: A, degree
        self.r_cut = r_cut;
        self.N_emb = N_emb;
        self.atom_max = atom_max;
        self.elements = elements
        # set neural network

        self.mlp_emb = nn.Sequential(
            nn.Linear(len(self.elements)**2, N_emb),
            nn.Sigmoid(),
            nn.Linear(N_emb, N_emb),
            nn.Tanh(),
            nn.Linear(N_emb, N_emb),
        )
        
        self.mlp_fit = nn.Sequential(
            nn.Linear(N_emb*int(N_emb//4), N_fit),
            nn.Tanh(),
            nn.Linear(N_fit,N_fit),
            nn.Sigmoid(),
            nn.Linear(N_fit,N_fit),
            nn.Tanh(),
            nn.Linear(N_fit, 1)
        )
        self.device = device;
        
    def s(self, r):
        if(r<1E-5):
            return 0;
        if(r<self.r_cut-0.2):
            return 1/r;
        elif(r<self.r_cut):
            return 1/2/r*(np.cos(np.pi*(r-self.r_cut+0.2)/0.2)+1);
        else:
            return 0;
        
    def G(self, S_vec):
        
        s = S_vec.shape;
        S_vec = S_vec.reshape([s[0]*s[1]*s[2],s[3]]);
        Gmat = self.mlp_emb(S_vec);

        return Gmat.reshape([s[0],s[1],s[2], self.N_emb]);
    
    def convert(self, atoms_l):
        
        Nmax = np.max([len(atoms.get_atomic_numbers()) for atoms in atoms_l]);
        self.R = torch.zeros([len(atoms_l), Nmax, self.atom_max, 4]).to(self.device);
        self.Sg = torch.zeros([len(atoms_l), Nmax, self.atom_max, len(self.elements)**2]).to(self.device);
        
        for ind, atoms in enumerate(atoms_l):
            atom_list = atoms.get_atomic_numbers();
            natm = len(atom_list);
            species   = list(set(atom_list));
            
            dist  = atoms.get_all_distances(mic=True)
            dist = [np.argwhere((d!=0)*(d<self.r_cut)).T[0] for d in dist];
            
            R = [atoms.get_distances(dist[j], j, mic=True, vector=True) for j in range(natm)];
            R = [[[1]+(1/norm(rij)*rij).tolist() for rij in R[i]]+[[0,0,0,0]]*(self.atom_max-len(R[i])) for i in range(len(R))];

            Smat = torch.tensor([[self.s(norm(rij)) for rij in Ri]+[0]*(self.atom_max-len(Ri)) for Ri in R],dtype=torch.float32).to(self.device); 
            
            R_out = torch.einsum('ij,ijk->ijk',[Smat,torch.tensor(R, dtype=torch.float32).to(self.device)])
            
            mask_species  = [];
            for d in range(len(dist)):
                term1 = [];
                for i in dist[d]:
                    term1.append([(atom_list[d]==s1 and atom_list[i]==s2) for s1, s2 in product(species,species)]);
                term2 = [[0]*len(species)**2]*(self.atom_max-len(dist[d]));
                mask_species.append( term1 + term2 );
            mask_species = torch.tensor(mask_species).to(self.device);
            Sg_out = torch.einsum('ij,ijk->ijk', [Smat,mask_species]);

            self.R[ind, :natm,:,:] = R_out;
            self.Sg[ind, :natm,:,:] = Sg_out;
            
    def forward(self, indl):
        
        Gmat = self.G(self.Sg[indl]);
        Dmat = torch.einsum('uijk,uijl,uiml,uimn->uikn',[Gmat[:,:,:,:int(self.N_emb//4)], self.R[indl], self.R[indl], Gmat]);

        s = Dmat.shape;
        D = Dmat.reshape([s[0],s[1],s[2]*s[3]]);
        Q = self.mlp_fit(D);
        Q = torch.mean(Q, axis=(1,2));
        
        return Q  #(torch.tanh(Q)+1)/2;
    
    def save(self, filename):
        torch.save(self.state_dict(), filename);
        
    def load(self, filename):
        self.load_state_dict(torch.load(filename))