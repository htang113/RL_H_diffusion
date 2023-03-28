import numpy as np;
import scipy;
import torch
from itertools import chain;
from scipy.sparse import linalg;

class descriptorNN(torch.nn.Module):
    def __init__(self, r_cut = 3.5, n_cut = 12, neuNum = 256):
        super(descriptorNN, self).__init__()
        self.name = 'descriptorNN';
        # set computation supercell box = [a, b, c, alpha, beta, gamma]. unit: A, degree
        self.r_cut = r_cut;
        self.n_cut = n_cut;
        # set neural network
        self.Linear1 = torch.nn.Linear(4*n_cut, neuNum);
        self.Linear2 = torch.nn.Linear(neuNum, neuNum);
        self.Linear3 = torch.nn.Linear(neuNum, 4);
        # Initialization of network paramenters; The elements of parameters should be tensor:
        #parameter[i][1] = torch,Tensor(out_features, in_features);
        #parameters[i][2] = torch.Tensor(out_features);

        torch.nn.init.kaiming_normal_(self.Linear1.weight, mode='fan_in', nonlinearity='relu');
        torch.nn.init.constant_(self.Linear1.bias.data, 0);
        torch.nn.init.kaiming_normal_(self.Linear2.weight, mode='fan_in', nonlinearity='relu');
        torch.nn.init.constant_(self.Linear2.bias.data, 0);
        torch.nn.init.xavier_normal_(self.Linear3.weight);
        torch.nn.init.constant_(self.Linear3.bias.data, 0);


            
    def convert(self,traj):
        natm = len(traj[0].numbers);
        pos_list = [at.get_positions() for at in traj];
        lat = traj[0].cell.tolist();
        lat_vec = np.array([lat[0][0],lat[1][1],lat[2][2]]);
        descriptor_pos = torch.zeros(len(pos_list),natm,4*self.n_cut);
        for p in range(len(pos_list)):
            pos = np.array(pos_list[p]);
            for i in range(natm):
                dis = pos-pos[i];
                tf1 = dis%lat_vec<self.r_cut;
                tf2 = (-dis)%lat_vec<self.r_cut;
                tf = tf1+tf2;
                tf = tf.astype(int);
                tf = np.sum(tf,axis=1);
                tf = tf>2;
                index = np.unique(np.nonzero(tf)[0]);
                shrink = dis[index,:];
                tf1 = shrink%lat_vec<self.r_cut;
                tf2 = (-shrink)%lat_vec<self.r_cut;
                shrink = tf1*(shrink%lat_vec) - tf2*((-shrink)%lat_vec);
                r = np.linalg.norm(shrink,axis=1);
                shrink = np.vstack((r,shrink.transpose()));
                # if we need it to be a sphere
                tf3 = r<self.r_cut;
                index = np.unique(np.nonzero(tf3)[0]);
                shrink = shrink[:,index];
                des = shrink;
                order = np.lexsort((des[3,:],des[2,:],des[1,:],des[0,:]));
                des = des.transpose();
                des = des[order,:];
                des_num = des.shape;
                if des_num[0]<self.n_cut+1:
                    EmptyPos = np.zeros(((self.n_cut+1-des_num[0]),4));
                    des = np.vstack((des,EmptyPos));
                des = des[1:self.n_cut+1,:];
                des = np.reshape(des,des.size);
                des = torch.tensor(des);
                descriptor_pos[p,i,:] = des;
        shape = descriptor_pos.shape;
        self.shape = shape;
        input_des = descriptor_pos.reshape(shape[0]*shape[1], shape[-1]);
        self.descript = input_des;
        return self.descript;
    
    def forward(self,descript):
        output_v = self.Linear1(descript);
        output_v = torch.nn.functional.relu(output_v);
        output_v = self.Linear2(output_v);
        output_v = torch.nn.functional.relu(output_v);
        output_v = self.Linear3(output_v);
        output_v0 = output_v[:,0];
        rest = [1,2,3];
        output_v1 = output_v[:,rest];
        v0_sorted = output_v0.reshape(self.shape[0], self.shape[1]);
        v1_sorted = output_v1.reshape(self.shape[0], self.shape[1], output_v1.shape[-1]);
        return [v0_sorted, v1_sorted];

        
        
        
        
    
