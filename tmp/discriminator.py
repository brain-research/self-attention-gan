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
import sn_ops as ops

slim = tf.contrib.slim


def dsample(x):
  xd = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
  return xd


def Block(x, out_channels, name, update_collection=tf.GraphKeys.UPDATE_OPS,
          downsample=True, act=tf.nn.relu):
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


def BaselineBlock(x, bottleneck_ratio, name, update_collection=None,
                  act=tf.nn.relu):
  with tf.variable_scope(name):
    input_channels = x.shape.as_list()[-1]
    out_channels = input_channels // bottleneck_ratio
    x_0 = x
    x = act(x)
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1,
                     update_collection=update_collection, name='sn_conv1')
    x = act(x)
    x = ops.snconv2d(x, input_channels, 3, 3, 1, 1,
                     update_collection=update_collection, name='sn_conv2')
    sigma = tf.get_variable(
        'sigma_ratio', [], initializer=tf.constant_initializer(0.0))
    return x_0 + sigma * x


def OptimizedBlock(x,  out_channels, name,
                   update_collection=tf.GraphKeys.UPDATE_OPS, act=tf.nn.relu):
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


def discriminator(image, labels, df_dim, number_classes, reuse_vars=False,
                  update_collection=tf.GraphKeys.UPDATE_OPS, act=tf.nn.relu):
  if reuse_vars:
    tf.get_variable_scope().reuse_variables()
  h0 = OptimizedBlock(image, df_dim, 'd_optimized_block1',
                      update_collection, act=act)
  h0 = BaselineBlock(h0, 8, 'd_baseline_block')
  h1 = Block(h0, df_dim * 2, 'd_block2', update_collection, act=act)
  h2 = Block(h1, df_dim * 4, 'd_block3', update_collection, act=act)
  h3 = Block(h2, df_dim * 8, 'd_block4', update_collection, act=act)
  h4 = Block(h3, df_dim * 16, 'd_block5', update_collection, act=act)
  h5 = Block(h4, df_dim * 16, 'd_block6', update_collection, False, act=act)
  h5_act = act(h5)
  h6 = tf.reduce_sum(h5_act, [1, 2])
  output = ops.snlinear(h6, 1, update_collection=update_collection,
                        name='d_sn_linear')
  h_labels = ops.sn_embedding(labels, number_classes, df_dim * 16,
                              update_collection=update_collection,
                              name='d_embedding')
  output += tf.reduce_sum(h6 * h_labels, axis=1, keepdims=True)
  return output


def discriminator_test(image, labels, df_dim, number_classes, reuse_vars=False,
                       update_collection=tf.GraphKeys.UPDATE_OPS,
                       act=tf.nn.relu):
  if reuse_vars:
    tf.get_variable_scope().reuse_variables()
  h0 = OptimizedBlock(image, df_dim, 'd_optimized_block1', update_collection,
                      act=act)
  h1 = Block(h0, df_dim * 2, 'd_block2', update_collection, act=act)
  h1 = non_local.sn_non_local_block_sim(h1, update_collection,
                                        name='d_non_local')
  h2 = Block(h1, df_dim * 4, 'd_block3', update_collection, act=act)
  h3 = Block(h2, df_dim * 8, 'd_block4', update_collection, act=act)
  h4 = Block(h3, df_dim * 16, 'd_block5', update_collection, act=act)
  h5 = Block(h4, df_dim * 16, 'd_block6', update_collection, False, act=act)
  h5_act = act(h5)
  h6 = tf.reduce_sum(h5_act, [1, 2])
  output = ops.snlinear(h6, 1, update_collection=update_collection,
                        name='d_sn_linear')
  h_labels = ops.sn_embedding(labels, number_classes, df_dim * 16,
                              update_collection=update_collection,
                              name='d_embedding')
  output += tf.reduce_sum(h6 * h_labels, axis=1, keepdims=True)
  return output
