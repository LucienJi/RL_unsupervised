import gym
from algo import SAC
from algo import generate_contrastive_example
env_name = 'BipedalWalker-v3'
env = gym.make(env_name)

with open('{}_test.csv'.format(env_name), 'w+') as myfile:
    myfile.write('{0},{1},{2}\n'.format("Episode", "Reward", "Value_Loss"))


print(env.action_space.shape)
print(env.observation_space.shape)
skill_dim =5
action_space = env.action_space

agent = SAC(env.observation_space,skill_dim,env.action_space)
total_episode = 10
per_episode = 500

for i_episode in range(total_episode):
    data = {}
    observation = env.reset()
    data['obs'] = observation.reshape(1,-1)
    done = False
    ct = 0
    agent.vae.eval()
    while ct < per_episode and not done:
        ct += 1
        #test = env.render(mode='rgb_array')
        env.render(mode='human')
        # 网络在每一个step都会根据 obs 输出 action，qvalue，policy
        # action 用于更新agent动作
        # qvalue，policy用于放入更新阶段

        data['act'] =action_space.sample()
        #print(data['act'])
        #print(np.shape(data['act']))
        data['next_obs'], external_reward, data['done'], info = env.step(data['act'])
        data['next_obs'] = data['next_obs'].reshape(1,-1)


        data['skill'] = agent.vae.sample_skill(data['obs'],data['next_obs'])
        data['skill_bar'] = generate_contrastive_example(data['skill'])

        data['rew'] = agent.vae.get_reward(data['obs'],data['next_obs'],data['skill'])

        agent.push(data)
        data['obs'] = data['next_obs']
        done = data['done']


    agent.vae.train()
    agent.train()


agent.save()
