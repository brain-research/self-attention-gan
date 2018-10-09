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

"""The generator of SNGAN."""

import tensorflow as tf
import ops
import non_local


def upscale(x, n):
  """Builds box upscaling (also called nearest neighbors).

  Args:
    x: 4D image tensor in B01C format.
    n: integer scale (must be a power of 2).

  Returns:
    4D tensor of images up scaled by a factor n.
  """
  if n == 1:
    return x
  return tf.batch_to_space(tf.tile(x, [n**2, 1, 1, 1]), [[0, 0], [0, 0]], n)


def usample_tpu(x):
  """Upscales the width and height of the input vector by a factor of 2."""
  x = upscale(x, 2)
  return x

def usample(x):
  _, nh, nw, nx = x.get_shape().as_list()
  x = tf.image.resize_nearest_neighbor(x, [nh * 2, nw * 2])
  return x

def block_no_sn(x, labels, out_channels, num_classes, is_training, name):
  """Builds the residual blocks used in the generator.

  Compared with block, optimized_block always downsamples the spatial resolution
  of the input vector by a factor of 4.

  Args:
    x: The 4D input vector.
    labels: The conditional labels in the generation.
    out_channels: Number of features in the output layer.
    num_classes: Number of classes in the labels.
    name: The variable scope name for the block.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.variable_scope(name):
    bn0 = ops.ConditionalBatchNorm(num_classes, name='cbn_0')
    bn1 = ops.ConditionalBatchNorm(num_classes, name='cbn_1')
    x_0 = x
    x = tf.nn.relu(bn0(x, labels, is_training))
    x = usample(x)
    x = ops.conv2d(x, out_channels, 3, 3, 1, 1, name='conv1')
    x = tf.nn.relu(bn1(x, labels, is_training))
    x = ops.conv2d(x, out_channels, 3, 3, 1, 1, name='conv2')

    x_0 = usample(x_0)
    x_0 = ops.conv2d(x_0, out_channels, 1, 1, 1, 1, name='conv3')

    return x_0 + x

def block(x, labels, out_channels, num_classes, is_training, name):
  with tf.variable_scope(name):
    bn0 = ops.ConditionalBatchNorm(num_classes, name='cbn_0')
    bn1 = ops.ConditionalBatchNorm(num_classes, name='cbn_1')
    x_0 = x
    x = tf.nn.relu(bn0(x, labels, is_training))
    x = usample(x)
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, name='snconv1')
    x = tf.nn.relu(bn1(x, labels, is_training))
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, name='snconv2')

    x_0 = usample(x_0)
    x_0 = ops.snconv2d(x_0, out_channels, 1, 1, 1, 1, name='snconv3')

    return x_0 + x


def generator_old(zs,
              target_class,
              gf_dim,
              num_classes,
              is_training=True,
              scope='Generator'):
  """Builds the generator graph propagating from z to x.

  Args:
    zs: The list of noise tensors.
    target_class: The conditional labels in the generation.
    gf_dim: The gf dimension.
    num_classes: Number of classes in the labels.
    scope: Optional scope for `variable_op_scope`.

  Returns:
    outputs: The output layer of the generator.
  """

  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    # project `z` and reshape
    act0 = ops.linear(zs, gf_dim * 16 * 4 * 4, scope='g_h0')
    act0 = tf.reshape(act0, [-1, 4, 4, gf_dim * 16])

    act1 = block_no_sn(act0, target_class, gf_dim * 16,
                 num_classes, is_training, 'g_block1')  # 8 * 8
    act2 = block_no_sn(act1, target_class, gf_dim * 8,
                 num_classes, is_training, 'g_block2')  # 16 * 16
    act3 = block_no_sn(act2, target_class, gf_dim * 4,
                 num_classes, is_training, 'g_block3')  # 32 * 32
    act4 = block_no_sn(act3, target_class, gf_dim * 2,
                 num_classes, is_training, 'g_block4')  # 64 * 64
    act5 = block_no_sn(act4, target_class, gf_dim,
                 num_classes, is_training, 'g_block5')  # 128 * 128
    bn = ops.batch_norm(name='g_bn')

    act5 = tf.nn.relu(bn(act5, is_training))
    act6 = ops.conv2d(act5, 3, 3, 3, 1, 1, name='g_conv_last')
    out = tf.nn.tanh(act6)
    print('GAN baseline with moving average')
    return out

def generator(zs,
               target_class,
               gf_dim,
               num_classes,
               is_training=True):
  """Builds the generator graph propagating from z to x.

  Args:
    zs: The list of noise tensors.
    target_class: The conditional labels in the generation.
    gf_dim: The gf dimension.
    num_classes: Number of classes in the labels.
    scope: Optional scope for `variable_op_scope`.

  Returns:
    outputs: The output layer of the generator.
  """

  with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    # project `z` and reshape
    act0 = ops.snlinear(zs, gf_dim * 16 * 4 * 4, name='g_snh0')
    act0 = tf.reshape(act0, [-1, 4, 4, gf_dim * 16])

    act1 = block(act0, target_class, gf_dim * 16,
                 num_classes, is_training, 'g_block1')  # 8 * 8
    act2 = block(act1, target_class, gf_dim * 8,
                 num_classes, is_training, 'g_block2')  # 16 * 16
    act3 = block(act2, target_class, gf_dim * 4,
                 num_classes, is_training, 'g_block3')  # 32 * 32
    act4 = block(act3, target_class, gf_dim * 2,
                 num_classes, is_training, 'g_block4')  # 64 * 64
    act5 = block(act4, target_class, gf_dim,
                 num_classes, is_training, 'g_block5')  # 128 * 128
    bn = ops.batch_norm(name='g_bn')

    act5 = tf.nn.relu(bn(act5, is_training))
    act6 = ops.snconv2d(act5, 3, 3, 3, 1, 1, name='g_snconv_last')
    out = tf.nn.tanh(act6)
    print('Generator Structure')
    return out


def generator_test(zs,
                   target_class,
                   gf_dim,
                   num_classes,
                   is_training=True):
  """Builds the generator graph propagating from z to x.

  Args:
    zs: The list of noise tensors.
    target_class: The conditional labels in the generation.
    gf_dim: The gf dimension.
    num_classes: Number of classes in the labels.
    scope: Optional scope for `variable_op_scope`.

  Returns:
    outputs: The output layer of the generator.
  """

  with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    # project `z` and reshape
    act0 = ops.snlinear(zs, gf_dim * 16 * 4 * 4, name='g_snh0')
    act0 = tf.reshape(act0, [-1, 4, 4, gf_dim * 16])

    act1 = block(act0, target_class, gf_dim * 16,
                 num_classes, is_training, 'g_block1')  # 8 * 8
    act2 = block(act1, target_class, gf_dim * 8,
                 num_classes, is_training, 'g_block2')  # 16 * 16
    act3 = block(act2, target_class, gf_dim * 4,
                 num_classes, is_training, 'g_block3')  # 32 * 32
    act3 = non_local.sn_non_local_block_sim(act3, None, name='g_non_local')
    act4 = block(act3, target_class, gf_dim * 2,
                 num_classes, is_training, 'g_block4')  # 64 * 64
    act5 = block(act4, target_class, gf_dim,
                 num_classes, is_training, 'g_block5')  # 128 * 128
    bn = ops.batch_norm(name='g_bn')

    act5 = tf.nn.relu(bn(act5, is_training))
    act6 = ops.snconv2d(act5, 3, 3, 3, 1, 1, name='g_snconv_last')
    out = tf.nn.tanh(act6)
    print('Generator TEST structure')
    return out

def generator_test_64(zs,
                   target_class,
                   gf_dim,
                   num_classes,
                   is_training=True):
  """Builds the generator graph propagating from z to x.

  Args:
    zs: The list of noise tensors.
    target_class: The conditional labels in the generation.
    gf_dim: The gf dimension.
    num_classes: Number of classes in the labels.
    scope: Optional scope for `variable_op_scope`.

  Returns:
    outputs: The output layer of the generator.
  """

  with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    # project `z` and reshape
    act0 = ops.snlinear(zs, gf_dim * 16 * 4 * 4, name='g_snh0')
    act0 = tf.reshape(act0, [-1, 4, 4, gf_dim * 16])

    act1 = block(act0, target_class, gf_dim * 16,
                 num_classes, is_training, 'g_block1')  # 8 * 8
    act2 = block(act1, target_class, gf_dim * 8,
                 num_classes, is_training, 'g_block2')  # 16 * 16
    act3 = block(act2, target_class, gf_dim * 4,
                 num_classes, is_training, 'g_block3')  # 32 * 32

    act4 = block(act3, target_class, gf_dim * 2,
                 num_classes, is_training, 'g_block4')  # 64 * 64
    act4 = non_local.sn_non_local_block_sim(act4, None, name='g_non_local')
    act5 = block(act4, target_class, gf_dim,
                 num_classes, is_training, 'g_block5')  # 128 * 128
    bn = ops.batch_norm(name='g_bn')

    act5 = tf.nn.relu(bn(act5, is_training))
    act6 = ops.snconv2d(act5, 3, 3, 3, 1, 1, name='g_snconv_last')
    out = tf.nn.tanh(act6)
    print('GAN test with moving average')
    return out

