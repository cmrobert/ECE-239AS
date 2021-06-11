import gym

from stable_baselines import SAC, PPO2
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import ACKTR

from gail.gail import GAIL
from gail.policy import MLPPolicy


env = gym.make("CartPole-v1")

# Generate expert trajectories (train expert)
# model = PPO2("MlpPolicy", env, verbose=1)
# generate_expert_traj(model, "expert_cartpole", n_timesteps=60000, n_episodes=10)

# Load the expert dataset
dataset = ExpertDataset(
    expert_path="expert_cartpole.npz", traj_limitation=10, verbose=1
)

model = GAIL(MLPPolicy, env, expert_dataset=dataset, verbose=1)
# Note: in practice, you need to train for 1M steps to have a working policy
model.learn(total_timesteps=1000)
model.save("gail_cartpole")

del model  # remove to demonstrate saving and loading

model = GAIL.load("gail_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
