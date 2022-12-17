import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))  # add parent directory

import gym
import gym_new_classic_envs

import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gym_new_classic_envs.utils.visualize import record_matplotlib_video

target = 0.0
env_id = "Arm-v0"
env = gym.make(env_id, target=target)
env = TimeLimit(env, max_episode_steps=200)
# env = make_vec_env(env, n_envs=1)

create_new = False

if create_new:
    # Instantiate the agent
    model = PPO(
        "MlpPolicy",
        env,
        gamma=0.98,
        # Using https://proceedings.mlr.press/v164/raffin22a.html
        use_sde=True,
        sde_sample_freq=4,
        learning_rate=1e-3,
        verbose=1,
    )
    model.learn(total_timesteps=1)
else:
    model = PPO.load("/tmp/gym/ppo_arm_0_2e5")



# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

record_matplotlib_video(env_id, model, target=target, video_length=1000, prefix='ppo-arm', video_folder='videos/')
exit()

model.learn(total_timesteps=int(2e5))

# Create save dir
save_dir = "/tmp/gym/"
os.makedirs(save_dir, exist_ok=True)
model.save(save_dir + "ppo_arm_0_2e5")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

record_matplotlib_video(env_id, model, target=target, video_length=1000, prefix='ppo-arm', video_folder='videos/')
