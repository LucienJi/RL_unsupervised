import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import copy


def generate_contrastive_example(z:np.ndarray):
    z = np.reshape(z,(-1,1))
    m  = m = np.matmul(z,z.transpose())/np.sum(z*z)

    r = np.random.rand((z.shape[0]),1)
    z_bar = r - np.matmul(m, r)

    z_bar /= np.linalg.norm(z_bar)

    return z_bar

def generate_sample_skill(z,length):
    """

    :param z: (1,latent)dim)
    :param length: int
    :return: (length,lat_dim)
    """
    sample_skill = np.zeros(shape=(length,z.shape[1]))
    for i in range(length):
        sample_skill[i] = generate_contrastive_example(z)
    return torch.from_numpy(sample_skill)

class BetaVAE(nn.Module):
    def __init__(self, obs_dim: int, lat_dim: int, hidden_dims: list,beta: float) ->None:
        super().__init__()
        self.lat_dim = lat_dim
        self.obs_dim = obs_dim
        self.beta = beta
        self.L = 20
        # encoder: (obs_dim,obs_dim,obs_dim)->lat_dim
        in_dim = 3*obs_dim
        module = []
        for h_dim in hidden_dims:
            module.append(
                nn.Sequential(
                    nn.Linear(in_dim,h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_dim = h_dim
        self.encoder = nn.Sequential(*module)
        self.fc_mu = nn.Linear(hidden_dims[-1],lat_dim)
        self.fc_var = nn.Linear(hidden_dims[-1],lat_dim)

        # decoder:(obs_dim,lat_dim)->obs_dim: delta_obs

        module = []
        self.decoder_input = nn.Linear(obs_dim + lat_dim,hidden_dims[-1])
        hidden_dims.reverse()
        for i in range(len(hidden_dims)-1):
            module.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(*module)
        self.final_layer_mu = nn.Linear(hidden_dims[-1],obs_dim)
        self.final_layer_var = nn.Linear(hidden_dims[-1],obs_dim)


    def encode(self,obs1,obs2):
        delta = obs2 - obs1
        input = torch.cat((obs1,obs2,delta),dim=-1)
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        log_var = F.softplus(log_var)
        log_var = torch.clamp(log_var,0.3,5)
        return [mu,log_var]

    def decode(self,obs1,z):
        input = torch.cat((obs1,z),dim=-1)
        result = self.decoder_input(input)
        result = self.decoder(result)
        delta_obs_mu = self.final_layer_mu(result)
        delta_obs_logvar = self.final_layer_var(result)
        delta_obs_logvar = F.softplus(delta_obs_logvar)
        delta_obs_logvar = torch.clamp(delta_obs_logvar,0.3,5)
        return [delta_obs_mu,delta_obs_logvar]

    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_skill(self,obs1,obs2):
        obs1 = torch.as_tensor(obs1)
        obs2 = torch.as_tensor(obs2)
        z = self.encode(obs1,obs2)
        z = self.reparameterize(z[0],z[1]).detach().numpy()
        return z
    def _generate_skill(self,obs1,obs2):
        z = self.encode(obs1, obs2)
        z = self.reparameterize(z[0], z[1])
        return z
    def _generate_skill_bar(self,z):
        """
        z:(batch,skill_dim)

        """
        batch_size = z.shape[0]
        z_norm = torch.sum(z*z,dim=-1)
        projection = torch.bmm(z.view(batch_size,-1,1),z.view(batch_size,1,-1))/z_norm.view(batch_size,1,1)
        z_bar = torch.rand_like(z)
        z_bar = z_bar.view(batch_size,-1,1) - torch.bmm(projection,z_bar.view(batch_size,-1,1))
        z_bar = z_bar.squeeze()
        z_bar /= torch.norm(z_bar,dim=-1).view(batch_size,1)
        return z_bar


    def reconstruction(self,obs1,obs2):
        mu,log_var = self.encode(obs1,obs2)
        z = self.reparameterize(mu,log_var)
        delta_mu,delta_log = self.decode(obs1,z)
        return delta_mu + obs1

    
    def _get_distribution(self,mu,logvar):
        
        std = torch.exp(logvar)
        distribution = torch.distributions.normal.Normal(loc=mu, scale=std)
        distribution = torch.distributions.Independent(distribution, 1)
        return distribution
    def _get_logp(self,distribution,realization):
        logp = distribution.log_prob(realization)
        return logp
    
    def _increase_prob(self,distribution,realization):
        logp = self._get_logp(distribution,realization)
        return -torch.mean(logp)
    
    def _decrease_prob(self,distribution,realization):
        logp = self._get_logp(distribution,realization)
        return torch.mean(logp)
    

    def loss_function(self,obs1,obs2,kld_weight):
        z_list = self.encode(obs1,obs2)
        mu = z_list[0]
        logvar = z_list[1]
        z = self.reparameterize(mu,logvar)
        # increase proba with obs
        delta_obs2 = self.decode(obs1, z)
        delta_obs2_mu = delta_obs2[0]
        delta_obs2_logvar = delta_obs2[1]
        recons = self._get_distribution(delta_obs2_mu + obs1,delta_obs2_logvar)
        loss1 = self._increase_prob(recons,obs2)

        # MSE loss
        mse_loss = F.mse_loss(delta_obs2_mu,obs2-obs1)
        
        # kl loss
        kld_loss= torch.mean(-0.5 * torch.sum(1 +delta_obs2_logvar - (delta_obs2_mu +obs1) ** 2 - delta_obs2_logvar.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss*self.beta*kld_weight
        loss = loss1 + mse_loss + kld_loss
        return {"MSE_Loss":mse_loss,
                "Prob_Loss":loss1,
                "KL_Loss":kld_loss}

    def contrastive_loss(self,obs1,obs2):
        z = self._generate_skill(obs1,obs2)
        z = z.detach()
        z_bar = self._generate_skill_bar(z)
        z_bar = z_bar.detach()
        ## generate Obs according to the z_bar
        delta_obs2_bar = self.decode(obs1,z_bar)
        delta_obs2_bar_mu = delta_obs2_bar[0]
        delta_obs2_bar_logvar = delta_obs2_bar[1]

        # recons
        z_bar_list = self.encode(obs1,delta_obs2_bar_mu+obs1)
        loss_mse = F.mse_loss(z_bar_list[0],z_bar)

        # decrease proba with obs
        reons = self._get_distribution(mu=delta_obs2_bar_mu+obs1,logvar=delta_obs2_bar_logvar)
        loss_decre1 = self._decrease_prob(distribution=reons,realization=obs2)
        loss_decre1 = -F.mse_loss(delta_obs2_bar_mu,obs2-obs1)

        # decrease proba with z and z_bar
        z = self.encode(obs1,obs2)
        z_recons = self._get_distribution(mu=z[0], logvar=z[1])
        loss_decre2 = self._decrease_prob(distribution=z_recons,realization=z_bar)

        return {"MSE_Loss": loss_mse,
                "Obs_P":loss_decre1,
                "Z_P":loss_decre2}

    def get_reward(self,obs1,obs2,z):
        """

        :param obs1: type numpy,
        :param obs2: type numpy
        :param z:  type numpy
        :return:  type double
        """
        L = 20
        sampled_skill = generate_sample_skill(z,L)

        obs1 = torch.as_tensor(obs1, dtype=torch.double)
        z = torch.as_tensor(z, dtype=torch.double)
        obs2 = torch.as_tensor(obs2, dtype=torch.double)

        input_obs_altz = torch.cat([obs1] * L, dim=0)
        target_obs_altz = torch.cat([obs2] * L, dim=0)

        # uniform = torch.distributions.uniform.Uniform(low=-1.0, high=1.0)
        # sampled_skill = uniform.rsample(sample_shape=(input_obs_altz.shape[0], self.skill_dim)).double()
        # print("sample_skill",sampled_skill)
        delta_obs2 = self.decode(obs1, z)
        delta_obs2_mu = delta_obs2[0]
        delta_obs2_logvar = delta_obs2[1]

        # increase proba with obs
        recons = self._get_distribution(mu=delta_obs2_mu + obs1, logvar=delta_obs2_logvar)
        log_pi= self._get_logp(distribution=recons, realization=obs2)

        altobs2 = self.decode(input_obs_altz,sampled_skill)
        altobs2 = self._get_distribution(mu=altobs2[0]+input_obs_altz,logvar=altobs2[1])
        logp_altz = self._get_logp(distribution=altobs2,realization=target_obs_altz)  # (batch_size,)
        logp_altz = logp_altz.detach().numpy()

        logp_altz = np.array(np.array_split(logp_altz, self.L))
        log_pi = log_pi.detach().numpy()
        intrinsic_reward = np.log(L + 1) - np.log(1 + np.exp(
            np.clip(logp_altz - log_pi.reshape(1, -1), -50, 50)).sum(axis=0))

        return intrinsic_reward





        











