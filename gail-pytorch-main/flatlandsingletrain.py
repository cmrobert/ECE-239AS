import os
import json
import pickle
import argparse
import matplotlib.pyplot as plt
import torch
import gym
import random
import sys
import numpy as np
from argparse import ArgumentParser, Namespace
from collections import deque
from pathlib import Path

from models.nets import Expert
from models.gail import GAIL

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from utils.observation_utils import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv


def main(env_name, gpu_num):
    
    expert_ckpt_path = "experts/Flatland/"

    print(expert_ckpt_path)
    with open(expert_ckpt_path + "model_config.json") as f:
        expert_config = json.load(f)

    ckpt_path = ".ckpts/Flatland/"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)["CartPole-v1"]

    with open(ckpt_path + "model_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    # Environment parameters
    n_agents = 1
    x_dim = 25
    y_dim = 25
    n_cities = 4
    max_rails_between_cities = 2
    max_rails_in_city = 3
    seed = 42

    # Observation parameters
    observation_tree_depth = 2
    observation_radius = 10

    # Exploration parameters
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.997  # for 2500ts

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)

    # Observation builder
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth)

    # Setup the environment
    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city
        ),
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=n_agents,
        obs_builder_object=tree_observation
    )

    env.reset(True, True)


    print(env.action_space)
    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = 0
    for i in range(observation_tree_depth + 1):
        n_nodes += np.power(4, i)
    state_size = n_features_per_node * n_nodes

    state_dim = state_size
    action_size = 5
    action_dim = 5

    # Max number of steps per episode
    # This is the official formula used during evaluations
    max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))

    action_dict = dict()

    # And some variables to keep track of the progress
    scores_window = deque(maxlen=100)  # todo smooth when rendering instead
    completion_window = deque(maxlen=100)
    scores = []
    completion = []
    action_count = [0] * action_size
    agent_obs = [None] * env.get_num_agents()
    agent_prev_obs = [None] * env.get_num_agents()
    agent_prev_action = [2] * env.get_num_agents()
    update_values = False


    
    
    


    
    
    ## Trying to load expert policy here
    
    
    
    expert = Expert(state_dim, action_dim, discrete = False, **expert_config)
    expert.pi.load_state_dict(torch.load(expert_ckpt_path + "sa-expert.pth"))

    model = GAIL(state_dim, action_dim, discrete, config)

    
    results = model.train(env, expert)
    print(results)
    env.close()

    with open(ckpt_path + "results.pkl", "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(model.pi.state_dict(), ckpt_path + "sa-expert.pth")
    if hasattr(model, "v"):
        torch.save(model.v.state_dict(), ckpt_path + "value.ckpt")
    if hasattr(model, "d"):
        torch.save(model.d.state_dict(), ckpt_path + "discriminator.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="Flatland",
        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3]"
    )
    parser.add_argument(
        "--gpu_num",
        type=int,
        default=0,
        help="Type the number of the GPU of your GPU mahine \
            you want to use if possible"
    )
    args = parser.parse_args()

    main(**vars(args))
