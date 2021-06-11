import gym

from stable_baselines import SAC
from stable_baselines.gail import ExpertDataset, generate_expert_traj

from gail.gail import GAIL
from gail.policy import MLPPolicy

# Load the expert dataset
dataset = ExpertDataset(
    expert_path="expert_pendulum.npz", traj_limitation=10, verbose=1
)

model = GAIL(MLPPolicy, "Pendulum-v0", expert_dataset=dataset, verbose=1)
model = GAIL.load("gail_pendulum")

env = gym.make("Pendulum-v0")
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
