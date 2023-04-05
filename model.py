import numpy as np;
from numpy.linalg import norm;
import scipy;
import torch
from itertools import chain;
from scipy.sparse import linalg;
from torch import nn;

class descriptorNN(torch.nn.Module):
    
    def __init__(self, elements=[1,29],r_cut = 3.5, n_cut = 12, N_emb=10, N_fit = 32, atom_max = 25):
        super(descriptorNN, self).__init__()
        
        self.name = 'descriptorNN';
        # set computation supercell box = [a, b, c, alpha, beta, gamma]. unit: A, degree
        self.r_cut = r_cut;
        self.n_cut = n_cut;
        self.N_emb = N_emb;
        self.atom_max = atom_max;
        self.elements = elements
        # set neural network

        self.mlp_emb = nn.Sequential(
            nn.Linear(len(self.elements)+1, N_emb),
            nn.ReLU(),
            nn.Linear(N_emb, N_emb),
        )
        
        self.mlp_fit = nn.Sequential(
            nn.Linear(N_emb*int(N_emb//4), N_fit),
            nn.ReLU(),
            nn.Linear(N_fit,N_fit),
            nn.ReLU(),
            nn.Linear(N_fit, 1)
        )
        
        self.mlp_val = nn.Sequential(
            nn.Linear(N_emb*int(N_emb//4), N_fit),
            nn.ReLU(),
            nn.Linear(N_fit,N_fit),
            nn.ReLU(),
            nn.Linear(N_fit, 1)
        )
        
        self.out_layer = nn.LogSoftmax();
    
    def s(self, r):
        
        if(r<self.r_cut-0.2):
            return 1/r;
        elif(r<self.r_cut):
            return 1/2/r*(np.cos(np.pi*(r-self.r_cut+0.2)/0.2)+1);
        else:
            return 0;
        
    def G(self, S_vec):
        
        s = S_vec.shape;
        S_vec = S_vec.reshape([s[0]*s[1],s[2]]);
        Gmat = self.mlp_emb(S_vec);

        return Gmat.reshape([s[0],s[1],self.N_emb]);
    
    def forward(self, atoms, actions):
        natm = len(atoms.numbers);
        H_ind = [a[0] for a in actions];
        pos_list = atoms.get_positions();
        atom_list = atoms.get_atomic_numbers();
        species   = ['action']+list(set(atom_list));
        
        dist  = [atoms.get_distances([q for q in range(natm)],j,mic=True) for j in H_ind];
        dist = [np.argwhere((d!=0)*(d<self.r_cut)).T[0] for d in dist];
        
        R = [pos_list[dist[j]]-pos_list[H_ind[j]] for j in range(len(H_ind))];
        
        Smat = torch.tensor([[1]+[self.s(norm(rij)) for rij in Ri]+[0]*(self.atom_max-len(Ri)) for Ri in R],dtype=torch.float32); 
        
        R = [[[1]+actions[i][1:]]+[[1]+(1/norm(rij)*rij).tolist() for rij in R[i]]+[[0,0,0,0]]*(self.atom_max-len(R[i])) for i in range(len(R))];
        R = torch.einsum('ij,ijk->ijk',[Smat,torch.tensor(R, dtype=torch.float32)]);
        
        mask_species = torch.tensor([[[True]+[False]*(len(species)-1)]+[[atom_list[i]==specie for specie in species] for i in dist[d]]+[[0]*len(species)]*(self.atom_max-len(dist[d])) for d in range(len(dist))]);
        Sg = torch.einsum('ij,ijk->ijk',[Smat,mask_species]);
        
        Gmat = self.G(Sg);
        Dmat = torch.einsum('ijk,ijl,iml,imn->ikn',[Gmat[:,:,:int(self.N_emb//4)], R, R, Gmat]);
        s = Dmat.shape;
        D = Dmat.reshape([s[0],s[1]*s[2]]);
        policy = self.out_layer(self.mlp_fit(D).flatten());
        
        Dval = torch.einsum('ijk,ijl,iml,imn->ikn',[Gmat[:,1:,:int(self.N_emb//4)], R[:,1:,:], R[:,1:,:], Gmat[:,1:,:]]);
        Dval = Dval.reshape([s[0],s[1]*s[2]]);
        value = torch.sum(self.mlp_val(Dval).flatten());
        
        return policy,value;
    
        
