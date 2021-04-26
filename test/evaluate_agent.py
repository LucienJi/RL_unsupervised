import gym
from algo import SAC
import numpy as np
env_name = 'BipedalWalker-v2'
env = gym.make(env_name)
print(env.action_space)
print(env.observation_space)
skill_dim =5

agent = SAC(env.observation_space,skill_dim,env.action_space)
agent.load()

total_episode = 1
per_episode = 2000

for i_episode in range(total_episode):
    data = {}
    observation = env.reset()
    data['obs'] = observation.reshape(1,-1)
    done = False
    ct = 0
    #data['skill'] = agent.sd.get_prior(data['obs'])
    data['skill'] = np.array([[0.,0.,0.,0.,1.]])
    print("Skill",data['skill'])
    while ct < per_episode and not done:
        ct += 1
        env.render()

        # 网络在每一个step都会根据 obs 输出 action，qvalue，policy
        # action 用于更新agent动作
        # qvalue，policy用于放入更新阶段

        data['act'] = agent.act(data['obs'],data["skill"])

        data['next_obs'], external_reward, data['done'], info = env.step(data['act'][0])
        data['next_obs'] = data['next_obs'].reshape(1,-1)
        data['rew'] = agent.sd.get_reward(data['obs'],data['skill'],data['next_obs'])

        agent.push(data)
        data['obs'] = data['next_obs']
        #done = data['done']


    #agent.train()
#agent.save()