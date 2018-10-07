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

"""The building block ops for Spectral Normalization GAN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from contextlib import contextmanager


rng = np.random.RandomState([2016, 6, 1])


def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=2, d_w=2, name='conv2d'):
  """Creates convolutional layers which use xavier initializer.

  Args:
    input_: 4D input tensor (batch size, height, width, channel).
    output_dim: Number of features in the output layer.
    k_h: The height of the convolutional kernel.
    k_w: The width of the convolutional kernel.
    d_h: The height stride of the convolutional kernel.
    d_w: The width stride of the convolutional kernel.
    name: The name of the variable scope.
  Returns:
    conv: The normalized tensor.
  """
  with tf.variable_scope(name):
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim],
                             initializer=tf.zeros_initializer())
    conv = tf.nn.bias_add(conv, biases)
    return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name='deconv2d', init_bias=0.):
  """Creates deconvolutional layers.

  Args:
    input_: 4D input tensor (batch size, height, width, channel).
    output_shape: Number of features in the output layer.
    k_h: The height of the convolutional kernel.
    k_w: The width of the convolutional kernel.
    d_h: The height stride of the convolutional kernel.
    d_w: The width stride of the convolutional kernel.
    stddev: The standard deviation for weights initializer.
    name: The name of the variable scope.
    init_bias: The initial bias for the layer.
  Returns:
    conv: The normalized tensor.
  """
  with tf.variable_scope(name):
    w = tf.get_variable('w',
                        [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                        initializer=tf.random_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])
    biases = tf.get_variable('biases', [output_shape[-1]],
                             initializer=tf.constant_initializer(init_bias))
    deconv = tf.nn.bias_add(deconv, biases)
    deconv.shape.assert_is_compatible_with(output_shape)

    return deconv


def linear(x, output_size, scope=None, bias_start=0.0):
  """Creates a linear layer.

  Args:
    x: 2D input tensor (batch size, features)
    output_size: Number of features in the output layer
    scope: Optional, variable scope to put the layer's parameters into
    bias_start: The bias parameters are initialized to this value

  Returns:
    The normalized tensor
  """
  shape = x.get_shape().as_list()

  with tf.variable_scope(scope or 'Linear'):
    matrix = tf.get_variable(
        'Matrix', [shape[1], output_size], tf.float32,
        tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable(
        'bias', [output_size], initializer=tf.constant_initializer(bias_start))
  out = tf.matmul(x, matrix) + bias
  return out


def lrelu(x, leak=0.2, name='lrelu'):
  """The leaky RELU operation."""
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def _l2normalize(v, eps=1e-12):
  """l2 normize the input vector."""
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(weights, num_iters=1, update_collection=None,
                           with_sigma=False):
  """Performs Spectral Normalization on a weight tensor.

  Specifically it divides the weight tensor by its largest singular value. This
  is intended to stabilize GAN training, by making the discriminator satisfy a
  local 1-Lipschitz constraint.
  Based on [Spectral Normalization for Generative Adversarial Networks][sn-gan]
  [sn-gan] https://openreview.net/pdf?id=B1QRgziT-

  Args:
    weights: The weight tensor which requires spectral normalization
    num_iters: Number of SN iterations.
    update_collection: The update collection for assigning persisted variable u.
                       If None, the function will update u during the forward
                       pass. Else if the update_collection equals 'NO_OPS', the
                       function will not update the u during the forward. This
                       is useful for the discriminator, since it does not update
                       u in the second pass.
                       Else, it will put the assignment in a collection
                       defined by the user. Then the user need to run the
                       assignment explicitly.
    with_sigma: For debugging purpose. If True, the fuction returns
                the estimated singular value for the weight tensor.
  Returns:
    w_bar: The normalized weight tensor
    sigma: The estimated singular value for the weight tensor.
  """
  w_shape = weights.shape.as_list()
  w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
  u = tf.get_variable('u', [1, w_shape[-1]],
                      initializer=tf.truncated_normal_initializer(),
                      trainable=False)
  u_ = u
  for _ in range(num_iters):
    v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
    u_ = _l2normalize(tf.matmul(v_, w_mat))

  sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
  w_mat /= sigma
  if update_collection is None:
    with tf.control_dependencies([u.assign(u_)]):
      w_bar = tf.reshape(w_mat, w_shape)
  else:
    w_bar = tf.reshape(w_mat, w_shape)
    if update_collection != 'NO_OPS':
      tf.add_to_collection(update_collection, u.assign(u_))
  if with_sigma:
    return w_bar, sigma
  else:
    return w_bar


def snconv2d(input_, output_dim,
             k_h=3, k_w=3, d_h=2, d_w=2,
             sn_iters=1, update_collection=None, name='snconv2d'):
  """Creates a spectral normalized (SN) convolutional layer.

  Args:
    input_: 4D input tensor (batch size, height, width, channel).
    output_dim: Number of features in the output layer.
    k_h: The height of the convolutional kernel.
    k_w: The width of the convolutional kernel.
    d_h: The height stride of the convolutional kernel.
    d_w: The width stride of the convolutional kernel.
    sn_iters: The number of SN iterations.
    update_collection: The update collection used in spectral_normed_weight.
    name: The name of the variable scope.
  Returns:
    conv: The normalized tensor.

  """
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


def snlinear(x, output_size, bias_start=0.0,
             sn_iters=1, update_collection=None, name='snlinear'):
  """Creates a spectral normalized linear layer.

  Args:
    x: 2D input tensor (batch size, features).
    output_size: Number of features in output of layer.
    bias_start: The bias parameters are initialized to this value
    sn_iters: Number of SN iterations.
    update_collection: The update collection used in spectral_normed_weight
    name: Optional, variable scope to put the layer's parameters into
  Returns:
    The normalized tensor
  """
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
                 update_collection=None, name='snembedding'):
  """Creates a spectral normalized embedding lookup layer.

  Args:
    x: 1D input tensor (batch size, ).
    number_classes: The number of classes.
    embedding_size: The length of the embeddding vector for each class.
    sn_iters: Number of SN iterations.
    update_collection: The update collection used in spectral_normed_weight
    name: Optional, variable scope to put the layer's parameters into
  Returns:
    The output tensor (batch size, embedding_size).
  """
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


class ConditionalBatchNorm_old(object):
  """Conditional BatchNorm.

  For each  class, it has a specific gamma and beta as normalization variable.
  """

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

class ConditionalBatchNorm(object):
  """Conditional BatchNorm.

  For each  class, it has a specific gamma and beta as normalization variable.
  """

  def __init__(self, num_categories, name='conditional_batch_norm', decay_rate=0.999, center=True,
               scale=True):
    with tf.variable_scope(name):
      self.name = name
      self.num_categories = num_categories
      self.center = center
      self.scale = scale
      self.decay_rate = decay_rate

  def __call__(self, inputs, labels, is_training=True):
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    axis = [0, 1, 2]
    shape = tf.TensorShape([self.num_categories]).concatenate(params_shape)
    moving_shape = tf.TensorShape([1, 1, 1]).concatenate(params_shape)

    with tf.variable_scope(self.name):
      self.gamma = tf.get_variable(
          'gamma', shape,
          initializer=tf.ones_initializer())
      self.beta = tf.get_variable(
          'beta', shape,
          initializer=tf.zeros_initializer())
      self.moving_mean = tf.get_variable('mean', moving_shape,
                          initializer=tf.zeros_initializer(),
                          trainable=False)
      self.moving_var = tf.get_variable('var', moving_shape,
                          initializer=tf.ones_initializer(),
                          trainable=False)

      beta = tf.gather(self.beta, labels)
      beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)
      gamma = tf.gather(self.gamma, labels)
      gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
      decay = self.decay_rate
      variance_epsilon = 1E-5
      if is_training:
        mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
        update_mean = tf.assign(self.moving_mean, self.moving_mean * decay + mean * (1 - decay))
        update_var = tf.assign(self.moving_var, self.moving_var * decay + variance * (1 - decay))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_var)
        #with tf.control_dependencies([update_mean, update_var]):
        outputs = tf.nn.batch_normalization(
            inputs, mean, variance, beta, gamma, variance_epsilon)
      else:
        outputs = tf.nn.batch_normalization(
              inputs, self.moving_mean, self.moving_var, beta, gamma, variance_epsilon)
      outputs.set_shape(inputs_shape)
      return outputs

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9999, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      # updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

class BatchNorm(object):
  """The Batch Normalization layer."""

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

@contextmanager
def variables_on_gpu0():
  old_fn = tf.get_variable
  def new_fn(*args, **kwargs):
    with tf.device('/gpu:0'):
      return old_fn(*args, **kwargs)
  tf.get_variable = new_fn
  yield
  tf.get_variable = old_fn


def avg_grads(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads