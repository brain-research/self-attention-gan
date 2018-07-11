# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

NO_OPS = 'NO_OPS'


def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(weights, u=None, num_iters=1, update_collection=None,
                           with_sigma=False):
  w_shape = weights.shape.as_list()
  w_mat = tf.reshape(weights, [-1, w_shape[-1]])
  if u is None:
    u = tf.get_variable('u', [1, w_shape[-1]],
                        initializer=tf.truncated_normal_initializer(),
                        trainable=False)
  _u = u
  for _ in range(num_iters):
    _v = _l2normalize(tf.matmul(_u, w_mat, transpose_b=True))
    _u = _l2normalize(tf.matmul(_v, w_mat))

  sigma = tf.squeeze(tf.matmul(tf.matmul(_v, w_mat), _u, transpose_b=True))
  w_mat = w_mat / sigma
  if update_collection is None:
    with tf.control_dependencies([u.assign(_u)]):
      w_bar = tf.reshape(w_mat, w_shape)
  else:
    w_bar = tf.reshape(w_mat, w_shape)
    if update_collection != NO_OPS:
      tf.add_to_collection(update_collection, u.assign(_u))
  if with_sigma:
    return w_bar, sigma
  else:
    return w_bar


def snconv2d(input_, output_dim,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
             sn_iters=1, update_collection=None, name='snconv2d'):
  with tf.variable_scope(name):
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=tf.contrib.layers.xavier_initializer())
    w_bar = spectral_normed_weight(w, num_iters=sn_iters,
                                   update_collection=update_collection)
    conv = tf.nn.conv2d(input_, w_bar, strides=[1, d_h, d_w, 1], padding='SAME')
    biases = tf.get_variable('biases', [output_dim],
                             initializer=tf.zeros_initializer())
    conv = tf.nn.bias_add(conv, biases)
    return conv


def snlinear(x, output_size, mean=0., stddev=0.02, bias_start=0.0,
             sn_iters=1, update_collection=None, name='snlinear'):
  shape = x.get_shape().as_list()

  with tf.variable_scope(name):
    matrix = tf.get_variable(
        'Matrix', [shape[1], output_size], tf.float32,
        tf.contrib.layers.xavier_initializer())
    matrix_bar = spectral_normed_weight(matrix, num_iters=sn_iters,
                                        update_collection=update_collection)
    bias = tf.get_variable(
        'bias', [output_size], initializer=tf.constant_initializer(bias_start))
    out = tf.matmul(x, matrix_bar) + bias
    return out


def sn_embedding(x, number_classes, embedding_size, sn_iters=1,
                 update_collection=None,
                 name='snembedding'):
  with tf.variable_scope(name):
    embedding_map = tf.get_variable(
        name='embedding_map',
        shape=[number_classes, embedding_size],
        initializer=tf.contrib.layers.xavier_initializer())
    embedding_map_bar_transpose = spectral_normed_weight(
        tf.transpose(embedding_map), num_iters=sn_iters,
        update_collection=update_collection)
    embedding_map_bar = tf.transpose(embedding_map_bar_transpose)
    return tf.nn.embedding_lookup(embedding_map_bar, x)


class ConditionalBatchNorm(object):

  def __init__(self, num_categories, name='conditional_batch_norm', center=True,
               scale=True):
    with tf.variable_scope(name):
      self.name = name
      self.num_categories = num_categories
      self.center = center
      self.scale = scale

  def __call__(self, inputs, labels):
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    axis = [0, 1, 2]
    shape = tf.TensorShape([self.num_categories]).concatenate(params_shape)

    with tf.variable_scope(self.name):
      self.gamma = tf.get_variable(
          'gamma', shape,
          initializer=tf.ones_initializer())
      self.beta = tf.get_variable(
          'beta', shape,
          initializer=tf.zeros_initializer())
      beta = tf.gather(self.beta, labels)
      beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)
      gamma = tf.gather(self.gamma, labels)
      gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
      mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
      variance_epsilon = 1E-5
      outputs = tf.nn.batch_normalization(
          inputs, mean, variance, beta, gamma, variance_epsilon)
      outputs.set_shape(inputs_shape)
      return outputs


class BatchNorm(object):

  def __init__(self, name='batch_norm', center=True,
               scale=True):
    with tf.variable_scope(name):
      self.name = name
      self.center = center
      self.scale = scale

  def __call__(self, inputs):
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape().as_list()
    params_shape = inputs_shape[-1]
    axis = [0, 1, 2]
    shape = tf.TensorShape([params_shape])
    with tf.variable_scope(self.name):
      self.gamma = tf.get_variable(
          'gamma', shape,
          initializer=tf.ones_initializer())
      self.beta = tf.get_variable(
          'beta', shape,
          initializer=tf.zeros_initializer())
      beta = self.beta
      gamma = self.gamma

      mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
      variance_epsilon = 1E-5
      outputs = tf.nn.batch_normalization(
          inputs, mean, variance, beta, gamma, variance_epsilon)
      outputs.set_shape(inputs_shape)
      return outputs
