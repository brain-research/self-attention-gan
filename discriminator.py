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
"""The discriminator of SNGAN."""

import tensorflow as tf
import ops
import non_local


def dsample(x):
  """Downsamples the image by a factor of 2."""

  xd = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
  return xd


def block(x, out_channels, name, update_collection=None,
          downsample=True, act=tf.nn.relu):
  """Builds the residual blocks used in the discriminator in SNGAN.

  Args:
    x: The 4D input vector.
    out_channels: Number of features in the output layer.
    name: The variable scope name for the block.
    update_collection: The update collections used in the
                       spectral_normed_weight.
    downsample: If True, downsample the spatial size the input tensor by
                a factor of 4. If False, the spatial size of the input tensor is
                unchanged.
    act: The activation function used in the block.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.variable_scope(name):
    input_channels = x.shape.as_list()[-1]
    x_0 = x
    x = act(x)
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1,
                     update_collection=update_collection, name='sn_conv1')
    x = act(x)
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1,
                     update_collection=update_collection, name='sn_conv2')
    if downsample:
      x = dsample(x)
    if downsample or input_channels != out_channels:
      x_0 = ops.snconv2d(x_0, out_channels, 1, 1, 1, 1,
                         update_collection=update_collection, name='sn_conv3')
      if downsample:
        x_0 = dsample(x_0)
    return x_0 + x


def optimized_block(x, out_channels, name,
                    update_collection=None, act=tf.nn.relu):
  """Builds the simplified residual blocks for downsampling.

  Compared with block, optimized_block always downsamples the spatial resolution
  of the input vector by a factor of 4.

  Args:
    x: The 4D input vector.
    out_channels: Number of features in the output layer.
    name: The variable scope name for the block.
    update_collection: The update collections used in the
                       spectral_normed_weight.
    act: The activation function used in the block.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.variable_scope(name):
    x_0 = x
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1,
                     update_collection=update_collection, name='sn_conv1')
    x = act(x)
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1,
                     update_collection=update_collection, name='sn_conv2')
    x = dsample(x)
    x_0 = dsample(x_0)
    x_0 = ops.snconv2d(x_0, out_channels, 1, 1, 1, 1,
                       update_collection=update_collection, name='sn_conv3')
    return x + x_0


def discriminator_old(image, labels, df_dim, number_classes, update_collection=None,
                  act=tf.nn.relu, scope='Discriminator'):
  """Builds the discriminator graph.

  Args:
    image: The current batch of images to classify as fake or real.
    labels: The corresponding labels for the images.
    df_dim: The df dimension.
    number_classes: The number of classes in the labels.
    update_collection: The update collections used in the
                       spectral_normed_weight.
    act: The activation function used in the discriminator.
    scope: Optional scope for `variable_op_scope`.
  Returns:
    A `Tensor` representing the logits of the discriminator.
  """
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    h0 = optimized_block(image, df_dim, 'd_optimized_block1',
                         update_collection, act=act)  # 64 * 64
    h1 = block(h0, df_dim * 2, 'd_block2',
               update_collection, act=act)  # 32 * 32
    h2 = block(h1, df_dim * 4, 'd_block3',
               update_collection, act=act)  # 16 * 16
    h3 = block(h2, df_dim * 8, 'd_block4', update_collection, act=act)  # 8 * 8
    h4 = block(h3, df_dim * 16, 'd_block5', update_collection, act=act)  # 4 * 4
    h5 = block(h4, df_dim * 16, 'd_block6', update_collection, False, act=act)
    h5_act = act(h5)
    h6 = tf.reduce_sum(h5_act, [1, 2])
    output = ops.snlinear(h6, 1, update_collection=update_collection,
                          name='d_sn_linear')
    h_labels = ops.sn_embedding(labels, number_classes, df_dim * 16,
                                update_collection=update_collection,
                                name='d_embedding')
    output += tf.reduce_sum(h6 * h_labels, axis=1, keepdims=True)
    return output


def discriminator(image, labels, df_dim, number_classes, update_collection=None,
                  act=tf.nn.relu):
  """Builds the discriminator graph.

  Args:
    image: The current batch of images to classify as fake or real.
    labels: The corresponding labels for the images.
    df_dim: The df dimension.
    number_classes: The number of classes in the labels.
    update_collection: The update collections used in the
                       spectral_normed_weight.
    act: The activation function used in the discriminator.
    scope: Optional scope for `variable_op_scope`.
  Returns:
    A `Tensor` representing the logits of the discriminator.
  """
  with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    h0 = optimized_block(image, df_dim, 'd_optimized_block1',
                         update_collection, act=act)  # 64 * 64
    h1 = block(h0, df_dim * 2, 'd_block2',
               update_collection, act=act)  # 32 * 32
    h2 = block(h1, df_dim * 4, 'd_block3',
               update_collection, act=act)  # 16 * 16
    h3 = block(h2, df_dim * 8, 'd_block4', update_collection, act=act)  # 8 * 8
    h4 = block(h3, df_dim * 16, 'd_block5', update_collection, act=act)  # 4 * 4
    h5 = block(h4, df_dim * 16, 'd_block6', update_collection, False, act=act)
    h5_act = act(h5)
    h6 = tf.reduce_sum(h5_act, [1, 2])
    output = ops.snlinear(h6, 1, update_collection=update_collection,
                          name='d_sn_linear')
    h_labels = ops.sn_embedding(labels, number_classes, df_dim * 16,
                                update_collection=update_collection,
                                name='d_embedding')
    output += tf.reduce_sum(h6 * h_labels, axis=1, keepdims=True)
    print('Discriminator Structure')
    return output

def discriminator_test(image, labels, df_dim, number_classes, update_collection=None,
                       act=tf.nn.relu):
  """Builds the discriminator graph.

  Args:
    image: The current batch of images to classify as fake or real.
    labels: The corresponding labels for the images.
    df_dim: The df dimension.
    number_classes: The number of classes in the labels.
    update_collection: The update collections used in the
                       spectral_normed_weight.
    act: The activation function used in the discriminator.
    scope: Optional scope for `variable_op_scope`.
  Returns:
    A `Tensor` representing the logits of the discriminator.
  """
  with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    h0 = optimized_block(image, df_dim, 'd_optimized_block1',
                         update_collection, act=act)  # 64 * 64
    h1 = block(h0, df_dim * 2, 'd_block2',
               update_collection, act=act)  # 32 * 32
    h1 = non_local.sn_non_local_block_sim(h1, update_collection, name='d_non_local')  # 32 * 32
    h2 = block(h1, df_dim * 4, 'd_block3',
               update_collection, act=act)  # 16 * 16
    h3 = block(h2, df_dim * 8, 'd_block4', update_collection, act=act)  # 8 * 8
    h4 = block(h3, df_dim * 16, 'd_block5', update_collection, act=act)  # 4 * 4
    h5 = block(h4, df_dim * 16, 'd_block6', update_collection, False, act=act)
    h5_act = act(h5)
    h6 = tf.reduce_sum(h5_act, [1, 2])
    output = ops.snlinear(h6, 1, update_collection=update_collection,
                          name='d_sn_linear')
    h_labels = ops.sn_embedding(labels, number_classes, df_dim * 16,
                                update_collection=update_collection,
                                name='d_embedding')
    output += tf.reduce_sum(h6 * h_labels, axis=1, keepdims=True)
    print('Discriminator Test Structure')
    return output

def discriminator_test_64(image, labels, df_dim, number_classes, update_collection=None,
                       act=tf.nn.relu):
  """Builds the discriminator graph.

  Args:
    image: The current batch of images to classify as fake or real.
    labels: The corresponding labels for the images.
    df_dim: The df dimension.
    number_classes: The number of classes in the labels.
    update_collection: The update collections used in the
                       spectral_normed_weight.
    act: The activation function used in the discriminator.
    scope: Optional scope for `variable_op_scope`.
  Returns:
    A `Tensor` representing the logits of the discriminator.
  """
  with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    h0 = optimized_block(image, df_dim, 'd_optimized_block1',
                         update_collection, act=act)  # 64 * 64
    h0 = non_local.sn_non_local_block_sim(h0, update_collection, name='d_non_local')  # 64 * 64
    h1 = block(h0, df_dim * 2, 'd_block2',
               update_collection, act=act)  # 32 * 32
    h2 = block(h1, df_dim * 4, 'd_block3',
               update_collection, act=act)  # 16 * 16
    h3 = block(h2, df_dim * 8, 'd_block4', update_collection, act=act)  # 8 * 8
    h4 = block(h3, df_dim * 16, 'd_block5', update_collection, act=act)  # 4 * 4
    h5 = block(h4, df_dim * 16, 'd_block6', update_collection, False, act=act)
    h5_act = act(h5)
    h6 = tf.reduce_sum(h5_act, [1, 2])
    output = ops.snlinear(h6, 1, update_collection=update_collection,
                          name='d_sn_linear')
    h_labels = ops.sn_embedding(labels, number_classes, df_dim * 16,
                                update_collection=update_collection,
                                name='d_embedding')
    output += tf.reduce_sum(h6 * h_labels, axis=1, keepdims=True)
    return output