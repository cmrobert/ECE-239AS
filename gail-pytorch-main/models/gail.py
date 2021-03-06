import numpy as np
import torch

from models.nets import PolicyNetwork, ValueNetwork, Discriminator
from utils.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch
from utils.observation_utils import normalize_observation

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

def rawRewardDict2vec(reward_dict, agent_idx):
    # Function to create a reward vector for a specific agent from a reward dictionary
    #   Used for rewards provided by Flatland (even for single agent)
    reward_vec = []
    len_seq = len(reward_dict)  # reward_dict is a list of dictionaries; need list of values
    for reward_i in range(len_seq):
        reward_vec.append(reward_dict[reward_i][agent_idx])
    return reward_vec


class GAIL:
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        self.v = ValueNetwork(self.state_dim)

        self.d = Discriminator(self.state_dim, self.action_dim, self.discrete)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            for net in self.get_networks():
                net.to(torch.device("cuda"))

        # Flatland Environment settings
        self.env_type = "flatland"
        self.regenerate_rail=True
        self.regenerate_schedule=True
        # Observation parameters
        self.observation_tree_depth = 2
        self.observation_radius = 10

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()

        if self.env_type == "flatland":
            #import pdb; pdb.set_trace()
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        state = FloatTensor(state)
        distb = self.pi(state)

        #action = distb.sample().detach().cpu().numpy()
        action = distb.sample().detach().cpu().numpy()
        action = np.argmax(action)

        return action

    def train(self, env, expert, render=False):
        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        horizon = self.train_config["horizon"]
        lambda_ = self.train_config["lambda"]
        gae_gamma = self.train_config["gae_gamma"]
        gae_lambda = self.train_config["gae_lambda"]
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        normalize_advantage = self.train_config["normalize_advantage"]

        # Flatland Environment parameters
        n_agents = 1
        x_dim = 25
        y_dim = 25
        n_cities = 4
        max_rails_between_cities = 2
        max_rails_in_city = 3
        seed = 42
        max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))

        agent_obs = [None] * env.get_num_agents()
        opt_d = torch.optim.Adam(self.d.parameters())

        exp_rwd_iter = []

        exp_obs = []
        exp_acts = []

        steps = 0

        print("\nStarting GAIL training...")
        while steps < num_steps_per_iter:
            ep_obs = []
            ep_rwds = []
            action_dict = dict()

            t = 0
            done = False

            if self.env_type == "flatland":
                ob_all, info = env.reset(self.regenerate_rail, self.regenerate_schedule)
                for agent in env.get_agent_handles():
                    if ob_all[agent]:
                        agent_obs[agent] = normalize_observation(ob_all[agent], self.observation_tree_depth,
                                                                 observation_radius=self.observation_radius)
                ob = agent_obs[agent]
            else:
                ob, info = env.reset()

            while not done and steps < num_steps_per_iter:
                if self.env_type == "flatland":
                    for agent in env.get_agent_handles():
                        if info['action_required'][agent]:
                            act = expert.act(agent_obs[agent])
                            action_dict.update({agent: act})
                            #import pdb; pdb.set_trace()
                else:
                    act = expert.act(ob)

                ep_obs.append(ob)
                exp_obs.append(ob)
                exp_acts.append(act)

                if render:
                    env.render()

                if self.env_type == "flatland":
                    ob, rwd, done, info = env.step(action_dict)
                else:
                    ob, rwd, done, info = env.step(act)

                ep_rwds.append(rwd)

                t += 1
                steps += 1

                if horizon is not None:
                    if t >= horizon:
                        break

            if done:
                exp_rwd_iter.append(np.sum(ep_rwds))

            ep_obs = FloatTensor(ep_obs)
            if self.env_type == "flatland":
                #import pdb; pdb.set_trace()
                ep_rwds = FloatTensor(rawRewardDict2vec(ep_rwds, 0))
            else:
                ep_rwds = FloatTensor(ep_rwds)

        exp_rwd_mean = np.mean(rawRewardDict2vec(exp_rwd_iter, 0))
        print(
            "Expert Reward Mean: {}".format(exp_rwd_mean)
        )

        exp_obs = FloatTensor(exp_obs)
        exp_acts = FloatTensor(np.array(exp_acts))

        rwd_iter_means = []
        for i in range(num_iters):
            rwd_iter = []

            obs = []
            acts = []
            rets = []
            advs = []
            gms = []

            steps = 0
            while steps < num_steps_per_iter:
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_costs = []
                ep_disc_costs = []
                ep_gms = []
                ep_lmbs = []

                t = 0
                done = False
                action_dict = dict()

                if self.env_type == "flatland":
                    ob_all, info = env.reset(self.regenerate_rail, self.regenerate_schedule)
                    for agent in env.get_agent_handles():
                        if ob_all[agent]:
                            agent_obs[agent] = normalize_observation(ob_all[agent], self.observation_tree_depth,
                                                                    observation_radius=self.observation_radius)
                    ob = agent_obs[agent]
                else:
                    ob = env.reset()

                while not done and steps < num_steps_per_iter:
                    if self.env_type == "flatland":
                        for agent in env.get_agent_handles():
                            if info['action_required'][agent]:
                                act = self.act(ob)
                                action_dict.update({agent: act})
                                #import pdb; pdb.set_trace()
                    else:
                        act = self.act(ob)

                    ep_obs.append(ob)
                    obs.append(ob)

                    ep_acts.append(act)
                    acts.append(act)

                    if render:
                        env.render()
                    if self.env_type == "flatland":
                        ob, rwd, done, info = env.step(action_dict)
                    else:
                        ob, rwd, done, info = env.step(act)

                    ep_rwds.append(rwd)
                    ep_gms.append(gae_gamma ** t)
                    ep_lmbs.append(gae_lambda ** t)

                    t += 1
                    steps += 1

                    if horizon is not None:
                        if t >= horizon:
                            break

                if done:
                    #import pdb; pdb.set_trace()
                    rwd_iter.append(np.sum(ep_rwds))

                ep_obs = FloatTensor(ep_obs)
                ep_acts = FloatTensor(np.array(ep_acts))
                if self.env_type == "flatland":
                    ep_rwds = FloatTensor(rawRewardDict2vec(ep_rwds, 0))
                else:
                    ep_rwds = FloatTensor(ep_rwds)
                # ep_disc_rwds = FloatTensor(ep_disc_rwds)
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)

                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts))\
                    .squeeze().detach()
                ep_disc_costs = ep_gms * ep_costs

                ep_disc_rets = FloatTensor(
                    [sum(ep_disc_costs[i:]) for i in range(t)]
                )
                ep_rets = ep_disc_rets / ep_gms

                rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat(
                    (self.v(ep_obs)[1:], FloatTensor([[0.]]))
                ).detach()
                ep_deltas = ep_costs.unsqueeze(-1)\
                    + gae_gamma * next_vals\
                    - curr_vals

                ep_advs = torch.FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                    for j in range(t)
                ])
                advs.append(ep_advs)

                gms.append(ep_gms)

            #import pdb; pdb.set_trace()
            #print(rwd_iter)
            rwd_iter = rawRewardDict2vec(rwd_iter, 0)
            rwd_iter_means.append(np.mean(rwd_iter))
            print(
                "Iterations: {},   Reward Mean: {}"
                .format(i + 1, np.mean(rwd_iter))
            )

            obs = FloatTensor(obs)
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms = torch.cat(gms)

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()
                
                
            if torch.cuda.is_available():
                advs.to(torch.device("cuda:0"))    
           
            #import pdb; pdb.set_trace()
            self.d.train()
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            nov_scores = self.d.get_logits(obs, acts)

            opt_d.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) \
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    nov_scores, torch.ones_like(nov_scores)
                )
            loss.backward()
            opt_d.step()

            self.v.train()
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(obs).detach()

            def constraint():
                return ((old_v - self.v(obs)) ** 2).mean()

            grad_diff = get_flat_grads(constraint(), self.v)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v)\
                    .detach()

                return hessian

            g = get_flat_grads(
                ((-1) * (self.v(obs).squeeze() - rets) ** 2).mean(), self.v
            ).detach()
            s = conjugate_gradient(Hv, g).detach()

            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))

            new_params = old_params + alpha * s

            set_params(self.v, new_params)

            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(obs)

            def L():
                distb = self.pi(obs)
                
                #import pdb; pdb.set_trace()
                acts_oh = torch.nn.functional.one_hot(acts.to(torch.int64), self.action_dim)
                return (advs.to(self.device) * torch.exp(
                            distb.log_prob(acts_oh)
                            - old_distb.log_prob(acts_oh).detach()
                        )).mean()

            def kld():
                distb = self.pi(obs)

                if self.discrete:
                    old_p = old_distb.probs.detach()
                    p = distb.probs

                    return (old_p * (torch.log(old_p) - torch.log(p)))\
                        .sum(-1)\
                        .mean()

                else:
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)

                    return (0.5) * (
                            (old_cov / cov).sum(-1)
                            + (((old_mean - mean) ** 2) / cov).sum(-1)
                            - self.action_dim
                            + torch.log(cov).sum(-1)
                            - torch.log(old_cov).sum(-1)
                        ).mean()

            grad_kld_old_param = get_flat_grads(kld(), self.pi)

            def Hv(v):
                hessian = get_flat_grads(
                    torch.dot(grad_kld_old_param, v),
                    self.pi
                ).detach()

                return hessian + cg_damping * v

            g = get_flat_grads(L(), self.pi).detach().to(self.device)

            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(
                g, s, Hs, max_kl, L, kld, old_params, self.pi
            )

            acts_oh = torch.nn.functional.one_hot(acts.to(torch.int64), self.action_dim)
            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts_oh))\
                .mean()
            grad_disc_causal_entropy = get_flat_grads(
                disc_causal_entropy, self.pi
            )
            new_params += lambda_ * grad_disc_causal_entropy

            set_params(self.pi, new_params)

        return exp_rwd_mean, rwd_iter_means
