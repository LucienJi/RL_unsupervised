from algo import *
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import random
import os
input_data_path = '../data/input.csv'
model_path = 'vae_model.pt'
f = open(input_data_path,'w')

def generate_data(obs_dim,num_data):
    init_obs = np.random.normal(loc=np.zeros(shape=(obs_dim,)))
    init_obs2 = np.random.normal(loc=np.zeros(shape=(obs_dim,)))

    A1 = np.random.randn(obs_dim,obs_dim)
    A1 += np.ones_like(A1)

    A2 = np.random.randn(obs_dim, obs_dim)
    A2 += np.ones_like(A2)

    A3 = np.random.randn(obs_dim, obs_dim)
    A3 += np.ones_like(A3)

    b = np.random.normal(loc=np.zeros(shape=(obs_dim,)))
    data = np.zeros(shape=(num_data,obs_dim))

    for i in range(num_data):
        data[i] = init_obs
        u = random.random()
        if u<0.33:
            init_obs = np.matmul(A1,init_obs)
        elif u < 0.67:
            init_obs = np.matmul(A2,init_obs)
        else:
            init_obs = np.matmul(A3,init_obs)

        init_obs/=np.linalg.norm(init_obs)

        #b = np.random.normal(loc=np.zeros(shape=(obs_dim,)))

    return data,A1,A2,A3,b

data,A1,A2,A3,b = generate_data(obs_dim=10,num_data=10240)
np.savetxt(input_data_path,data,delimiter=',')

obs1 = torch.from_numpy(data[0:-1,:]).double()
obs2 = torch.from_numpy(data[1:,:]).double()

vae = BetaVAE(obs_dim=10,lat_dim=3,hidden_dims=[24,12,6],beta=1).double()



def train_vae_1(model:BetaVAE,obs1,obs2,num_epoch=5000):
    for i in range(num_epoch):
        paras = model.parameters()
        optm = Adam(paras, lr=0.001)
        loss = model.loss_function(obs1, obs2, 0.01*(num_epoch - i))
        optm.zero_grad()
        mse_loss,kl_loss,p_loss = loss['MSE_Loss'],loss['KL_Loss'],loss['Prob_Loss']
        loss = mse_loss + kl_loss + p_loss
        new_loss = loss
        new_loss.backward()
        optm.step()
        if i%100 == 0:
            print(i,"  ",mse_loss.data.item())

    torch.save(model.state_dict(),model_path)

def train_vae_2(model:BetaVAE,obs1,obs2,num_epoch=5000):
    for i in range(num_epoch):
        paras = model.parameters()
        optm = Adam(paras, lr=0.001)
        loss = model.loss_function(obs1, obs2, 0.1*(num_epoch - i))
        c_loss = model.contrastive_loss(obs1,obs2)
        optm.zero_grad()
        mse_loss,kl_loss,p_loss = loss['MSE_Loss'],loss['KL_Loss'],loss['Prob_Loss']
        mse2_loss,obs_p,z_p = c_loss['MSE_Loss'],c_loss['Obs_P'],c_loss['Z_P']

        new_loss =mse_loss  + p_loss +  (mse2_loss + z_p) * 0.01*(num_epoch - i)
        new_loss = mse_loss + p_loss
        new_loss.backward()
        optm.step()
        if i%100 == 0:
            print(i,"  ",mse_loss.data.item())
def test_my_vae(model:BetaVAE,obs1,obs2):
    recons = model.reconstruction(obs1,obs2)
    error = obs2 - recons #(num,dim)
    error = torch.norm(error)
    print(error.data.item())
    return obs2 - recons



train_vae_2(vae,obs1,obs2)
#vae.load_state_dict(torch.load(model_path))
#vae.eval()
error = test_my_vae(vae,obs1,obs2)
recons = error.detach().numpy()
#f1 = open('../data/output.csv','w')
np.savetxt('../data/output.csv',recons)
