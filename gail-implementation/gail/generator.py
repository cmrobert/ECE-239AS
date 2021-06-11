import tensorflow as tf

from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.common import tf_util as tf_util


class Generator:
    def __init__(
        self,
        scope,
        obs_rms,
        observation_shape,
        n_actions,
        discrete_actions,
        normalize,
        hidden_size,
    ):
        self.scope = scope
        self.obs_rms = obs_rms
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.discrete_actions = discrete_actions
        self.normalize = normalize
        self.hidden_size = hidden_size

    def model(self, obs_ph, acs_ph, reuse):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if self.normalize:
                with tf.variable_scope("obfilter"):
                    self.obs_rms = RunningMeanStd(shape=self.observation_shape)
                obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            else:
                obs = obs_ph

            if self.discrete_actions:
                one_hot_actions = tf.one_hot(acs_ph, self.n_actions)
                actions_ph = tf.cast(one_hot_actions, tf.float32)
            else:
                actions_ph = acs_ph

            _input = tf.concat(
                [obs, actions_ph], axis=1
            )  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(
                _input, self.hidden_size, activation_fn=tf.nn.tanh
            )
            p_h2 = tf.contrib.layers.fully_connected(
                p_h1, self.hidden_size, activation_fn=tf.nn.tanh
            )
            logits = tf.contrib.layers.fully_connected(
                p_h2, 1, activation_fn=tf.identity
            )
        return logits
