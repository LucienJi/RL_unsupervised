from torch.optim import Adam
import copy
from algo.Skill_learner.vae import BetaVAE
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
import itertools


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, skill_dim, act_dim, size):
        self.obs_buf = np.zeros(shape=combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(shape=combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(shape=combined_shape(size, act_dim), dtype=np.float32)
        self.skill_buff = np.zeros(shape=(size, skill_dim))
        self.skill_bar = np.zeros(shape= (size,skill_dim),dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, skill,skill_bar, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.skill_buff[self.ptr] = skill
        self.skill_bar[self.ptr] = skill_bar
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     next_obs=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     skill=self.skill_buff[idxs],
                     skill_bar = self.skill_bar[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.double) for k, v in batch.items()}


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation=activation, output_activation=nn.Identity)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, skill, deterministic=False, with_logprob=True):
        obs = torch.cat((obs, skill), dim=-1)
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()

        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, skill, act):
        obs = torch.cat((obs, skill), dim=-1)
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, skill_dim, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        skill_dim = skill_dim
        obs_dim = obs_dim + skill_dim

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, skill, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, skill, deterministic, False)
            return a.numpy()



class SAC:
    def __init__(self, obs_dim, skill_dim, act_dim, batch_size=256, learning_rate=1e-3, discount_coef=0.99,
                 entropy_coef=0.1,
                 update_coef=0.995):
        self.gamma = discount_coef
        self.alpha = entropy_coef
        self.polyak = update_coef

        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        self.act_dim = act_dim

        self.buffer = ReplayBuffer(self.obs_dim.shape, self.skill_dim, self.act_dim.shape, size=2 ** 12)
        self.iteration_ct = 0
        self.batch_size = batch_size

        ###
        #self.sd = SKillDiscriminator(obs_dim.shape[0], skill_dim).double()
        hidden_dims = [36,24,12]
        self.vae = BetaVAE(obs_dim = obs_dim.shape[0], lat_dim=skill_dim, hidden_dims=hidden_dims,beta=0.4).double()
        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(observation_space=obs_dim, skill_dim=skill_dim, action_space=act_dim,
                                 hidden_sizes=(32, 32), activation=nn.ReLU).double()
        self.ac_targ = copy.deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_params = self.ac.pi.parameters()
        self.vae_params = self.vae.parameters()

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=learning_rate)
        self.q_optimizer = Adam(self.q_params, lr=learning_rate)
        self.vae_optimizer = Adam(self.vae_params, lr=learning_rate)

    def _loss(self, data):
        obs = data['obs']
        r = data['rew']
        act = data['act']
        skill = data['skill']
        skill_bar = data['skill_bar']
        obs2 = data['next_obs']
        done = data['done']

        q1 = self.ac.q1(obs, skill, act)
        q2 = self.ac.q2(obs, skill, act)
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(obs2, skill)
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(obs2, skill, a2)
            q2_pi_targ = self.ac_targ.q2(obs2, skill, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - done) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        pi, logp_pi = self.ac.pi(obs, skill)

        q1_pi = self.ac.q1(obs, skill, pi)
        q2_pi = self.ac.q2(obs, skill, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        loss_vae = self.vae.loss_function(obs,obs2,0.4)
        loss_contrastive = self.vae.contrastive_loss(obs,obs2,skill_bar)


        return loss_pi, loss_q, loss_vae,loss_contrastive

    def update(self, data):

        loss_list = self._loss(data)
        loss_pi = loss_list[0]
        loss_q = loss_list[1]
        loss_vae = loss_list[2]['VAE_Loss']
        loss_contrastive = loss_list[3]['Contrastive_Loss']
        """ SD update"""
        self.vae_optimizer.zero_grad()
        loss_vae = loss_vae + loss_contrastive
        loss_vae.backward()
        self.vae_optimizer.step()

        """ Q value update"""
        for paras in self.ac.pi.parameters():
            paras.requires_grad = False

        self.q_optimizer.zero_grad()
        loss_q = loss_q
        loss_q.backward()
        self.q_optimizer.step()

        for paras in self.ac.pi.parameters():
            paras.requires_grad = True

        """ Policy update"""
        for paras in self.q_params:
            paras.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi = loss_pi
        loss_pi.backward()
        self.pi_optimizer.step()

        for paras in self.q_params:
            paras.requires_grad = True

        """ Update """
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data = (self.polyak) * p_targ.data + ((1 - self.polyak) * p.data)

        return loss_q.detach(), loss_pi.detach(), loss_vae

    def act(self, obs, skill):
        action = self.ac.act(torch.as_tensor(obs, dtype=torch.double), torch.as_tensor(skill, dtype=torch.double))
        return action

    def random_act(self):
        action = np.random.random(size = (self.act_dim.shape[0],))*self.act_dim.high[0]
        return action

    def push(self, data):
        self.buffer.store(obs=data['obs'],
                          act=data['act'],
                          rew=data['rew'],
                          skill=data['skill'],
                          skill_bar = data['skill_bar'],
                          next_obs=data['next_obs'],
                          done=data['done'])

    def train(self):
        self.iteration_ct += 1

        data = self.buffer.sample_batch(self.batch_size)
        loss_q, loss_pi, loss_sd = self.update(data)
        print("## Iteration {0}: Q LOSS {1}".format(self.iteration_ct, loss_q))

    def save(self, path=None):
        if path == None:
            torch.save(self.ac.state_dict(), "data/paras.pkl")

        else:
            torch.save(self.ac.state_dict(), path)

        print("Saved")

    def load(self, path=None):
        if path == None:
            self.ac.load_state_dict(torch.load("data/paras.pkl"))
        else:
            self.ac.load_state_dict(torch.load(path))
            self.ac.eval()

        print("Loaded")
