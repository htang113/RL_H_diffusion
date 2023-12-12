import numpy as np;
from numpy.linalg import norm;
import scipy;
import torch
from itertools import chain;
from itertools import product;
from scipy.sparse import linalg;
from torch import nn;
        
class Q_NN(torch.nn.Module):
    
    def __init__(self, elements=[1,29],r_cut = 3.5, N_emb=10, N_fit = 32, atom_max = 30):
        super(Q_NN, self).__init__()
        
        self.name = 'descriptorNN';
        # set computation supercell box = [a, b, c, alpha, beta, gamma]. unit: A, degree
        self.r_cut = r_cut;
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
            nn.Linear(N_fit, 2)
        )
        
    def s(self, r):
        
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

        return Gmat.reshape([s[0],s[1],s[2],self.N_emb]);
    
    def convert(self, atoms, actions):
        nframes = len(atoms); 
        natm = len(atoms[0].numbers);
        H_ind = [[a[0] for a in act] for act in actions];
        pos_list = [atom.get_positions() for atom in atoms];
        atom_list = atoms[0].get_atomic_numbers();
        species   = ['action']+list(set(atom_list));
        
        dist  = [[atoms[i].get_distances([q for q in range(natm)],j,mic=True) for j in H_ind[i]] for i in range(nframes)];
        dist = [[np.argwhere((d!=0)*(d<self.r_cut)).T[0] for d in di] for di in dist];
        
        R = [[atoms[i].get_distances(dist[i][j], H_ind[i][j], mic=True, vector=True) for j in range(len(H_ind[i]))] for i in range(nframes)];
        
        Smat = torch.tensor([[[1]+[self.s(norm(rij)) for rij in Ri]+[0]*(self.atom_max-len(Ri)) for Ri in RI] for RI in R],dtype=torch.float32); 
        
        R = [[[[1]+actions[u][i][1:]]+[[1]+(1/norm(rij)*rij).tolist() for rij in R[u][i]]+[[0,0,0,0]]*(self.atom_max-len(R[u][i])) for i in range(len(R[u]))] for u in range(nframes)];
        self.R = torch.einsum('uij,uijk->uijk',[Smat,torch.tensor(R, dtype=torch.float32)]);
        
        mask_species = torch.tensor([[[[True]+[False]*(len(species)-1)]+[[atom_list[i]==specie for specie in species] for i in dist[I][d]]+[[0]*len(species)]*(self.atom_max-len(dist[I][d])) for d in range(len(dist[I]))] for I in range(nframes)]);
        self.Sg = torch.einsum('uij,uijk->uijk',[Smat,mask_species]);
        
    def forward(self):
        
        Gmat = self.G(self.Sg);
        Dmat = torch.einsum('uijk,uijl,uiml,uimn->uikn',[Gmat[:,:,:,:int(self.N_emb//4)], self.R, self.R, Gmat]);
        s = Dmat.shape;
        D = Dmat.reshape([s[0]*s[1],s[2]*s[3]]);
        Q = self.mlp_fit(D);
        return Q.reshape([s[0],s[1],2]);
    
    def save(self, filename):
        torch.save(self.state_dict(), filename);
        
    def load(self, filename):
        self.load_state_dict(torch.load(filename))

        
class DQN(torch.nn.Module):
    
    def __init__(self, elements=[1,29],r_cut = 3.5, N_emb=10, N_fit = 32, atom_max = 30):
        super(DQN, self).__init__()
        
        self.name = 'descriptorNN';
        # set computation supercell box = [a, b, c, alpha, beta, gamma]. unit: A, degree
        self.r_cut = r_cut;
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
        
    def s(self, r):
        
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

        return Gmat.reshape([s[0],s[1],s[2],self.N_emb]);
    
    def convert(self, atoms, actions):
        nframes = len(atoms); 
        natm = len(atoms[0].numbers);
        H_ind = [[a[0] for a in act] for act in actions];
        pos_list = [atom.get_positions() for atom in atoms];
        atom_list = atoms[0].get_atomic_numbers();
        species   = ['action']+list(set(atom_list));
        
        dist  = [[atoms[i].get_distances([q for q in range(natm)],j,mic=True) for j in H_ind[i]] for i in range(nframes)];
        dist = [[np.argwhere((d!=0)*(d<self.r_cut)).T[0] for d in di] for di in dist];
        
        R = [[atoms[i].get_distances(H_ind[i][j], dist[i][j], mic=True, vector=True) for j in range(len(H_ind[i]))] for i in range(nframes)];
        
        Smat = torch.tensor([[[1]+[self.s(norm(rij)) for rij in Ri]+[0]*(self.atom_max-len(Ri)) for Ri in RI] for RI in R],dtype=torch.float32); 
        
        R = [[[[1]+actions[u][i][1:]]+[[1]+(1/norm(rij)*rij).tolist() for rij in R[u][i]]+[[0,0,0,0]]*(self.atom_max-len(R[u][i])) for i in range(len(R[u]))] for u in range(nframes)];
        self.R = torch.einsum('uij,uijk->uijk',[Smat,torch.tensor(R, dtype=torch.float32)]);
        
        mask_species = torch.tensor([[[[True]+[False]*(len(species)-1)]+[[atom_list[i]==specie for specie in species] for i in dist[I][d]]+[[0]*len(species)]*(self.atom_max-len(dist[I][d])) for d in range(len(dist[I]))] for I in range(nframes)]);
        self.Sg = torch.einsum('uij,uijk->uijk',[Smat,mask_species]);
       
    def forward(self):
        
        Gmat = self.G(self.Sg);
        Dmat = torch.einsum('uijk,uijl,uiml,uimn->uikn',[Gmat[:,:,:,:int(self.N_emb//4)], self.R, self.R, Gmat]);
        s = Dmat.shape;
        D = Dmat.reshape([s[0]*s[1],s[2]*s[3]]);
        Q = self.mlp_fit(D);
        return Q.reshape([s[0],s[1],1]);
    
    def save(self, filename):
        torch.save(self.state_dict(), filename);
        
    def load(self, filename):
        self.load_state_dict(torch.load(filename));

class DQN2(torch.nn.Module):
    
    def __init__(self, elements=[1,29],r_cut = 3.5, N_emb=10, N_fit = 32, atom_max = 30):
        super(DQN2, self).__init__()
        
        self.name = 'descriptorNN';
        # set computation supercell box = [a, b, c, alpha, beta, gamma]. unit: A, degree
        self.r_cut = r_cut;
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
        
    def s(self, r):
        
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

        return Gmat.reshape([s[0],s[1],s[2],self.N_emb]);
    
    def convert(self, atoms, actions):
        nframes = len(atoms); 
        natm = len(atoms[0].numbers);
        H_ind = [[a[0] for a in act] for act in actions];
        pos_list = [atom.get_positions() for atom in atoms];
        atom_list = atoms[0].get_atomic_numbers();
        species   = ['action']+list(set(atom_list));
        latt = atoms[0].get_cell();
        dist  = [[atoms[i].get_distances([q for q in range(natm)],j,mic=True) for j in H_ind[i]] for i in range(nframes)];
        dist = [[np.argwhere((d!=0)*(d<self.r_cut)).T[0] for d in di] for di in dist];
        R = [];
        a = []
        for i in range(nframes):
            R.append([]);
            a.append([]);
            for j in range(len(H_ind[i])):
                R[-1].append([]);
                a[-1].append([]);
                Rij = atoms[i].get_distances(H_ind[i][j], dist[i][j], mic=True, vector=True);
                for q in range(len(Rij)):
                    r = Rij[q];
                    for A in [[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1]]:
                        res = r+latt[0]*A[0]+latt[1]*A[1];
                        if(np.linalg.norm(res)<self.r_cut):
                            R[-1][-1].append(res);
                            a[-1][-1].append(atom_list[dist[i][j][q]])
                            
        Smat = torch.tensor([[[1]+[self.s(norm(rij)) for rij in Ri]+[0]*(self.atom_max-len(Ri)) for Ri in RI] for RI in R],dtype=torch.float32); 
        
        R = [[[[1]+actions[u][i][1:]]+[[1]+(1/norm(rij)*rij).tolist() for rij in R[u][i]]+[[0,0,0,0]]*(self.atom_max-len(R[u][i])) for i in range(len(R[u]))] for u in range(nframes)];
        self.R = torch.einsum('uij,uijk->uijk',[Smat,torch.tensor(R, dtype=torch.float32)]);
        
        mask_species = torch.tensor([[[[True]+[False]*(len(species)-1)]+[[a2==specie for specie in species] for a2 in a1]+[[0]*len(species)]*(self.atom_max-len(a1)) for a1 in a[I]] for I in range(nframes)]);
        self.Sg = torch.einsum('uij,uijk->uijk',[Smat,mask_species]);
       
    def forward(self):
        
        Gmat = self.G(self.Sg);
        Dmat = torch.einsum('uijk,uijl,uiml,uimn->uikn',[Gmat[:,:,:,:int(self.N_emb//4)], self.R, self.R, Gmat]);
        s = Dmat.shape;
        D = Dmat.reshape([s[0]*s[1],s[2]*s[3]]);
        Q = self.mlp_fit(D);
        return Q.reshape([s[0],s[1],1]);
    
    def save(self, filename):
        torch.save(self.state_dict(), filename);
        
    def load(self, filename):
        self.load_state_dict(torch.load(filename));