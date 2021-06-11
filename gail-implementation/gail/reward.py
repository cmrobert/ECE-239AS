import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.common import tf_util as tf_util

from .discriminator import Discriminator
from .generator import Generator
from .utils import *


class RewardClassifier(object):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        entcoeff=0.001,
        scope="adversary",
        normalize=True,
    ):
        self.scope = scope
        self.observation_shape = observation_space.shape
        self.actions_shape = action_space.shape

        if isinstance(action_space, gym.spaces.Box):
            # Continuous action space
            self.discrete_actions = False
            self.n_actions = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Discrete):
            self.n_actions = action_space.n
            self.discrete_actions = True
        else:
            raise ValueError("Action space not supported: {}".format(action_space))

        self.hidden_size = hidden_size
        self.normalize = normalize
        self.obs_rms = None

        # Placeholders
        self.generator_obs_ph = tf.placeholder(
            observation_space.dtype,
            (None,) + self.observation_shape,
            name="observations_ph",
        )
        self.generator_acs_ph = tf.placeholder(
            action_space.dtype, (None,) + self.actions_shape, name="actions_ph"
        )
        self.expert_obs_ph = tf.placeholder(
            observation_space.dtype,
            (None,) + self.observation_shape,
            name="expert_observations_ph",
        )
        self.expert_acs_ph = tf.placeholder(
            action_space.dtype, (None,) + self.actions_shape, name="expert_actions_ph"
        )

        generator_logits = Generator(
            self.scope,
            self.obs_rms,
            self.observation_shape,
            self.n_actions,
            self.discrete_actions,
            self.normalize,
            self.hidden_size,
        ).model(self.generator_obs_ph, self.generator_acs_ph, reuse=False)

        discriminator = Discriminator(
            self.scope,
            self.obs_rms,
            self.observation_shape,
            self.n_actions,
            self.discrete_actions,
            self.normalize,
            self.hidden_size,
        )
        discriminator_logits = discriminator.model(
            self.expert_obs_ph, self.expert_acs_ph, reuse=True
        )
        self.obs_rms = discriminator.obs_rms

        # Build accuracy
        generator_acc = tf.reduce_mean(
            tf.cast(tf.nn.sigmoid(generator_logits) < 0.5, tf.float32)
        )
        expert_acc = tf.reduce_mean(
            tf.cast(tf.nn.sigmoid(discriminator_logits) > 0.5, tf.float32)
        )
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=generator_logits, labels=tf.zeros_like(generator_logits)
        )
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=discriminator_logits, labels=tf.ones_like(discriminator_logits)
        )
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, discriminator_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy
        # Loss + Accuracy terms
        self.losses = [
            generator_loss,
            expert_loss,
            entropy,
            entropy_loss,
            generator_acc,
            expert_acc,
        ]
        self.loss_name = [
            "generator_loss",
            "expert_loss",
            "entropy",
            "entropy_loss",
            "generator_acc",
            "expert_acc",
        ]
        self.total_loss = generator_loss + expert_loss + entropy_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        var_list = self.get_trainable_variables()
        self.lossandgrad = tf_util.function(
            [
                self.generator_obs_ph,
                self.generator_acs_ph,
                self.expert_obs_ph,
                self.expert_acs_ph,
            ],
            self.losses + [tf_util.flatgrad(self.total_loss, var_list)],
        )

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, actions):
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 0)
        elif len(actions.shape) == 0:
            # one discrete action
            actions = np.expand_dims(actions, 0)

        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: actions}
        reward = sess.run(self.reward_op, feed_dict)
        return reward
