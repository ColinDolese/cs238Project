import gym
from Agent import DQNAgent
import numpy as np

games = 1000

env = gym.make('Centipede-ram-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
# agent.load("./save/cartpole-dqn.h5")
env._max_episode_steps = None
done = False
batch_size = 1000
for e in range(games):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    time = 0
    done = False
    while not done:
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        agent.short_replay(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, games, time, agent.epsilon))
            break
        # if len(agent.memory) > batch_size:
        #     agent.replay(batch_size)
        time += 1
    agent.replay(batch_size)
    if agent.epsilon > agent.epsilon_min:
    	agent.epsilon *= agent.epsilon_decay


# for i_episoderange(1):
#     observation = env.reset()
#     for t in range(10000):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print observation
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break


