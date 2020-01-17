import gym
import numpy as np
import tensorflow as tf
from stable_baselines.a2c.utils import conv_to_fc, linear, ortho_init
from stable_baselines.common.policies import ActorCriticPolicy

from config import INSTRUMENTS, INTERVALS, N_BATCHES, WINDOW


def reshape(original):
    tensor = tf.transpose(original, [0, 2, 4, 3, 1])
    tensor = tf.reshape(
        tensor, [-1, len(INTERVALS)*7, WINDOW, len(INSTRUMENTS)])
    tensor = tf.gather(tensor, indices=[3, 4, 6], axis=2)
    return tensor


def extractor(data, **kwargs):
    data = reshape(data)
    activ = tf.nn.relu

    with tf.variable_scope('c1'):
        norm1 = tf.keras.layers.BatchNormalization()
        depthwise_filter_1 = tf.get_variable('depthwise_filter_1', [len(INTERVALS)*3, 5, len(
            INSTRUMENTS), 3], initializer=ortho_init(1.0), dtype=tf.float32)
        # pointwise_filter_1 = tf.get_variable('pointwise_filter_1', [1, 1, len(
        #    INSTRUMENTS), 32], initializer=ortho_init(1.0), dtype=tf.float32)
        layer_1 = activ(norm1(tf.nn.depthwise_conv2d(data, filter=depthwise_filter_1,
                                               strides=[1, 1, 1, 1], padding='SAME', **kwargs)))

    with tf.variable_scope('c2'):
        norm2 = tf.keras.layers.BatchNormalization()
        filter_2 = tf.get_variable('filter_2', [len(
            INTERVALS)*3, 5, 3*5, 32], initializer=ortho_init(1.0))
        layer_2 = activ(norm2(tf.nn.conv2d(layer_1, filter=filter_2, strides=[
                        1, 1, 1, 1], padding='SAME', **kwargs)))

    with tf.variable_scope('c3'):
        norm3 = tf.keras.layers.BatchNormalization()
        filter_3 = tf.get_variable('filter_3', [len(
            INTERVALS)*3, 3, 32, 64], initializer=ortho_init(1.0))
        layer_3 = activ(norm3(tf.nn.conv2d(layer_2, filter=filter_3, strides=[
                        1, 1, 1, 1], padding='SAME', **kwargs)))

    with tf.variable_scope('fc'):
        layer_3 = conv_to_fc(layer_2)
        output = activ(
            linear(layer_3, 'fc', n_hidden=256, init_scale=np.sqrt(2)))

    return output


class A2CPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch=32, reuse=False, **kwargs):
        super(A2CPolicy, self).__init__(sess, ob_space, ac_space,
                                        n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu
            extracted_features = extractor(self.processed_obs, **kwargs)
            extracted_features = tf.layers.flatten(extracted_features)

            pi_h = extracted_features
            for i, layer_size in enumerate([128, 128]):
                pi_h = activ(tf.layers.dense(
                    pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = extracted_features
            for i, layer_size in enumerate([32, 32]):
                vf_h = activ(tf.layers.dense(
                    vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(
                    pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
