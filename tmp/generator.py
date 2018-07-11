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
import non_local
import sn_ops


def usample(x):
  _, nh, nw, _ = x.get_shape().as_list()
  x = tf.image.resize_nearest_neighbor(x, [nh * 2, nw * 2])
  return x


def Block(x, labels, out_channels, num_classes, name):
  with tf.variable_scope(name):
    bn0 = sn_ops.ConditionalBatchNorm(num_classes, name='cbn_0')
    bn1 = sn_ops.ConditionalBatchNorm(num_classes, name='cbn_1')
    x_0 = x
    x = tf.nn.relu(bn0(x, labels))
    x = usample(x)
    x = sn_ops.snconv2d(x, out_channels, 3, 3, 1, 1, name='snconv1')
    x = tf.nn.relu(bn1(x, labels))
    x = sn_ops.snconv2d(x, out_channels, 3, 3, 1, 1, name='snconv2')

    x_0 = usample(x_0)
    x_0 = sn_ops.snconv2d(x_0, out_channels, 1, 1, 1, 1, name='snconv3')

    return x_0 + x


def BaselineBlock(x, bottleneck_ratio, name, update_collection=None,
                  act=tf.nn.relu):
  with tf.variable_scope(name):
    input_channels = x.shape.as_list()[-1]
    out_channels = input_channels // bottleneck_ratio
    x_0 = x
    x = act(x)
    x = sn_ops.snconv2d(x, out_channels, 3, 3, 1, 1,
                        update_collection=update_collection, name='sn_conv1')
    x = act(x)
    x = sn_ops.snconv2d(x, input_channels, 3, 3, 1, 1,
                        update_collection=update_collection, name='sn_conv2')
    sigma = tf.get_variable(
        'sigma_ratio', [], initializer=tf.constant_initializer(0.0))
    return x_0 + sigma * x


def generator(zs,
              target_class,
              gf_dim,
              num_classes,
              reuse_vars=False):

  if reuse_vars:
    tf.get_variable_scope().reuse_variables()

  act0 = sn_ops.snlinear(zs, gf_dim * 16 * 4 * 4, name='g_snh0')
  act0 = tf.reshape(act0, [-1, 4, 4, gf_dim * 16])

  act1 = Block(act0, target_class, gf_dim * 16, num_classes, 'g_block1')
  act2 = Block(act1, target_class, gf_dim * 8, num_classes, 'g_block2')
  act3 = Block(act2, target_class, gf_dim * 4, num_classes, 'g_block3')
  act4 = Block(act3, target_class, gf_dim * 2, num_classes, 'g_block4')
  act4 = BaselineBlock(act4, 8, 'g_baseline_block')
  act5 = Block(act4, target_class, gf_dim, num_classes, 'g_block5')
  bn = sn_ops.BatchNorm(name='g_bn')

  act5 = tf.nn.relu(bn(act5))
  act6 = sn_ops.snconv2d(act5, 3, 3, 3, 1, 1, name='g_snconv_last')
  out = tf.nn.tanh(act6)
  return out


def generator_test(zs,
                   target_class,
                   gf_dim,
                   num_classes,
                   reuse_vars=False):

  if reuse_vars:
    tf.get_variable_scope().reuse_variables()

  act0 = sn_ops.snlinear(zs, gf_dim * 16 * 4 * 4, name='g_snh0')
  act0 = tf.reshape(act0, [-1, 4, 4, gf_dim * 16])
  act1 = Block(act0, target_class, gf_dim * 16, num_classes, 'g_block1')
  act2 = Block(act1, target_class, gf_dim * 8, num_classes, 'g_block2')
  act3 = Block(act2, target_class, gf_dim * 4, num_classes, 'g_block3')
  act3 = non_local.sn_non_local_block_sim(act3, None, name='g_non_local')
  act4 = Block(act3, target_class, gf_dim * 2, num_classes, 'g_block4')
  act5 = Block(act4, target_class, gf_dim, num_classes, 'g_block5')
  bn = sn_ops.BatchNorm(name='g_bn')

  act5 = tf.nn.relu(bn(act5))
  act6 = sn_ops.snconv2d(act5, 3, 3, 3, 1, 1, name='g_snconv_last')
  out = tf.nn.tanh(act6)
  return out
