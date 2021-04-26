from algo import *
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from numpy import genfromtxt
import matplotlib.pyplot as plt

vae = BetaVAE(obs_dim=10,lat_dim=3,hidden_dims=[24,12,6],beta=1).double()
in_dir = '../data/input.csv'
model_dir = 'vae_model.pt'
vae.load_state_dict(torch.load(model_dir))
obs = genfromtxt(in_dir,delimiter=',')
print(obs.shape)

obs1 = obs[0:-1,:]
obs2 = obs[1:,:]

z = vae.sample_skill(obs1,obs2)
print(z.shape)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(z[:,0],z[:,1],z[:,2])
plt.show()