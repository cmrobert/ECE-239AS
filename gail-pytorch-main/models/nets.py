import torch
from argparse import Namespace
import numpy as np
from reinforcement_learning.dddqn_policy import DDDQNPolicy

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, discrete):
        super(PolicyNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, action_dim),
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if not self.discrete:
            self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def forward(self, states):
        if self.discrete:
            probs = torch.nn.functional.softmax(self.net(states))
            distb = torch.distributions.Categorical(probs)
        else:
            mean = self.net(states)

            std = torch.exp(self.log_std)
            cov_mtx = torch.eye(self.action_dim) * (std ** 2)

            distb = torch.distributions.MultivariateNormal(mean, cov_mtx)

        return distb


class ValueNetwork(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 1),
        )

    def forward(self, states):
        return self.net(states)


class Discriminator(torch.nn.Module):
    def __init__(self, state_dim, action_dim, discrete):
        super(Discriminator, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if self.discrete:
            self.act_emb = torch.nn.Embedding(
                action_dim, state_dim
            )
            self.net_in_dim = 2 * state_dim
        else:
            self.net_in_dim = state_dim + action_dim

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.net_in_dim, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 1),
        )

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        #import pdb; pdb.set_trace()
        # Actions need to be in one-hot format (not index format)
        actions_oh = torch.nn.functional.one_hot(actions.to(torch.int64), self.action_dim)

        if self.discrete:
            actions = self.act_emb(actions.long())

        sa = torch.cat([states, actions_oh], dim=-1)

        return self.net(sa)


class Expert:
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None,
        expert_algo="dddqn"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config
        self.expert_algo = expert_algo

        # Determine the device to use
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Select the expert policy algorithm (current options listed below):
        if self.expert_algo == "dddqn":
            # Duelling Double Deep-Q Network
            training_parameters = {            # Training parameters
                'buffer_size': int(1e5),
                'batch_size': 32,
                'update_every': 8,
                'learning_rate': 0.5e-4,
                'tau': 1e-3,
                'gamma': 0.99,
                'buffer_min_size': 0,
                'hidden_size': 256,
                'use_gpu': True
            }
            # Actual policy
            self.pi = DDDQNPolicy(self.state_dim, self.action_dim, 
                                  Namespace(**training_parameters), evaluation_mode=False) #evaluation_mode=True)
        elif self.expert_algo == "default":
            # Fully connected neural network policy method
            self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)

        if torch.cuda.is_available():
            for net in self.get_networks():
                net.to(torch.device("cuda"))

    def get_networks(self):
        return [self.pi]

    def act(self, state):
        if self.expert_algo == "dddqn":
            self.pi.qnetwork_local.eval()
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            distb = self.pi.qnetwork_local(state)
            action = np.argmax(distb.cpu().data.numpy())
        elif self.expert_algo == "default":
            self.pi.eval()
            state = FloatTensor(state)
            distb = self.pi(state)

        

            action = distb.sample().detach().cpu().numpy()

        return action
