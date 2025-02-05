"""
Testing the trained checkpoint in a completely clean environment.

Using the original hockey env: pip install git+https://github.com/martius-lab/hockey-env.git
and sbx: pip install sbx-rl
"""

import numpy as np
from hockey.hockey_env import BasicOpponent, HockeyEnv
from sbx import CrossQ

env = HockeyEnv()
model = CrossQ.load("models/crossq/crossq_niclas.zip")
opponent = BasicOpponent(weak=False)

n_episodes = 100
rewards = np.full((n_episodes, env.max_timesteps), np.nan)
success = np.full(n_episodes, False)

for episode in range(n_episodes):
    obs1, info = env.reset()
    done = False

    for step in range(env.max_timesteps):
        env.render()
        obs2 = env.obs_agent_two()
        a1, _ = model.predict(obs1)
        a2 = opponent.act(obs2)
        obs1, reward, done, _, info = env.step(np.hstack([a1, a2]))
        rewards[episode, step] = reward

        if done:
            break

    success[episode] = info["winner"] == 1
    print(f"{info['winner']=}")

mean_reward = np.mean(np.nansum(rewards, axis=1))
mean_episode_length = np.mean(np.sum(~np.isnan(rewards), axis=1))
success_rate = np.mean(success)

print(f"{mean_reward=}")
print(f"{mean_episode_length=}")
print(f"{success_rate=}")
