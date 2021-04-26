import torch
import torch.nn as nn
from . import tools
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SKillDiscriminator(nn.Module):
    def __init__(self, obs_dim, skill_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        self.L = skill_dim # Because of discrete skill
        """supposing the prior distribution of skill is uniform in [-1,1] for each dim
        """

        out_dim = obs_dim

        self.net = tools.mlp(obs_dim + skill_dim,out_dim,n_layer=2)

        self.mean = nn.Sequential(
            nn.Linear(in_features=out_dim, out_features=obs_dim)
        )
        self.log_std = nn.Sequential(
            nn.Linear(in_features=out_dim, out_features=obs_dim),
            nn.Softplus()
        )

    def _skill_prior(self):
        """:arg
        suppose skill distribution: discrete equal proba on [0,1]
        """
        #prior = torch.distributions.multinomial.Multinomial(total_count=1, logits=torch.tensor([1.0] * self.skill_dim))

        prior = torch.distributions.one_hot_categorical.OneHotCategorical(torch.tensor([1.0/self.skill_dim]*self.skill_dim))

        return prior

    def get_prior(self, obs):
        prior = self._skill_prior()
        sz = obs.shape[0]
        skills = prior.sample(sample_shape=(sz,)).double()
        return skills.detach().numpy()

    def get_distribution(self, obs, skill):
        input = torch.cat((obs,skill),dim=-1)
        #obs = self.obs_net(obs)
        #skill = self.skill_net(skill)  # (batch_size,out_dim)

        out = self.net(input)  # (batch_size, out_dim*2)
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        distribution = torch.distributions.normal.Normal(loc=mean, scale=std)
        distribution = torch.distributions.Independent(distribution, 1)
        return distribution

    def get_logp(self, obs, skill, next_obs):
        distribution = self.get_distribution(obs, skill)
        logp = distribution.log_prob(next_obs)
        return logp

    def predict_obs(self, obs, skill):
        obs_ = self.obs_net(obs)
        skill_ = self.skill_net(skill)  # (batch_size,out_dim)

        out = torch.cat((obs_, skill_), dim=-1)  # (batch_size, out_dim*2)
        mean = self.mean(out)
        next_state = mean + obs
        return next_state.double()

    def decrease_logp(self, obs, skill, next_obs):
        logp = self.get_logp(obs, skill, next_obs)  # (batch_size,)
        return torch.mean(logp)

    def increase_log(self, obs, skill, next_obs):
        logp = self.get_logp(obs, skill, next_obs)
        return -torch.mean(logp)

    def get_reward(self, obs, skill, next_obs):
        """:arg
        obs: tensor (batch_size,obs_dim)
        skill: tensor(batch_size,skill_dim)
        next_obs: tensor (batch_size,obs_dim)

        """
        sampled_skill = np.concatenate([np.roll(skill, i, axis=1) for i in range(0, self.skill_dim)],axis=0)
        obs = torch.as_tensor(obs,dtype=torch.double)
        skill = torch.as_tensor(skill,dtype=torch.double)
        next_obs = torch.as_tensor(next_obs,dtype=torch.double)

        input_obs_altz = torch.cat([obs] * self.L, dim=0)
        target_obs_altz = torch.cat([next_obs] * self.L, dim=0)

        #uniform = torch.distributions.uniform.Uniform(low=-1.0, high=1.0)
        #sampled_skill = uniform.rsample(sample_shape=(input_obs_altz.shape[0], self.skill_dim)).double()
        #print("sample_skill",sampled_skill)
        log_pi = self.get_logp(obs, skill, next_obs)  # (batch_size,)
        log_pi = log_pi.detach().numpy()

        logp_altz = self.get_logp(input_obs_altz, torch.as_tensor(sampled_skill), target_obs_altz)  # (batch_size,)
        logp_altz = logp_altz.detach().numpy()

        logp_altz = np.array(np.array_split(logp_altz, self.L))

        intrinsic_reward = np.log(self.L + 1) - np.log(1 + np.exp(
            np.clip(logp_altz - log_pi.reshape(1, -1), -50, 50)).sum(axis=0))

        return intrinsic_reward



