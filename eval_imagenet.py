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

"""Generic train."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
import tensorflow as tf


import generator as generator_module
import utils_ori as utils




slim = tf.contrib.slim
tfgan = tf.contrib.gan


flags.DEFINE_string(
    # 'data_dir', '/gpu/hz138/Data/imagenet',  #'/home/hz138/Data/imagenet',
    'data_dir', '/bigdata1/hz138/Data/imagenet',
    'Directory with Imagenet input data as sharded recordio files of pre-'
    'processed images.')
flags.DEFINE_integer('z_dim', 128, 'The dimension of z')
flags.DEFINE_integer('gf_dim', 64, 'Dimensionality of gf. [64]')


flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use')
flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'Directory name to load '
                    'the checkpoints. [checkpoint]')
flags.DEFINE_string('sample_dir', 'sample', 'Directory name to save the '
                    'image samples. [sample]')
flags.DEFINE_string('eval_dir', 'checkpoint/eval', 'Directory name to save the '
                    'eval summaries . [eval]')
flags.DEFINE_integer('batch_size', 64, 'Batch size of samples to feed into '
                     'Inception models for evaluation. [16]')
flags.DEFINE_integer('shuffle_buffer_size', 5000, 'Number of records to load '
                     'before shuffling and yielding for consumption. [5000]')
flags.DEFINE_integer('dcgan_generator_batch_size', 100, 'Size of batch to feed '
                     'into generator -- we may stack multiple of these later.')
flags.DEFINE_integer('eval_sample_size', 50000,
                     'Number of samples to sample from '
                     'generator and real data. [1024]')
flags.DEFINE_boolean('is_train', False, 'Use DCGAN only for evaluation.')

flags.DEFINE_integer('task', 0, 'The task id of the current worker. [0]')
flags.DEFINE_integer('ps_tasks', 0, 'The number of ps tasks. [0]')
flags.DEFINE_integer('num_workers', 1, 'The number of worker tasks. [1]')
flags.DEFINE_integer('replicas_to_aggregate', 1, 'The number of replicas '
                     'to aggregate for synchronous optimization [1]')

flags.DEFINE_integer('num_towers', 1, 'The number of GPUs to use per task. [1]')
flags.DEFINE_integer('eval_interval_secs', 300,
                     'Frequency of generator evaluation with Inception score '
                     'and Frechet Inception Distance. [300]')

flags.DEFINE_integer('num_classes', 1000, 'The number of classes in the dataset')
flags.DEFINE_string('generator_type', 'test', 'test or baseline')

FLAGS = flags.FLAGS


def main(_):
  model_dir = '%s_%s' % ('imagenet', FLAGS.batch_size)
  FLAGS.eval_dir = FLAGS.checkpoint_dir + '/eval'
  checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
  log_dir = os.path.join(FLAGS.eval_dir, model_dir)
  print('log_dir', log_dir)
  graph_def = None  # pylint: disable=protected-access

  # Batch size to feed batches of images through Inception and the generator
  # to extract feature vectors to later stack together and compute metrics.
  local_batch_size = FLAGS.dcgan_generator_batch_size
  if FLAGS.generator_type == 'baseline':
    generator_fn = generator_module.generator
  elif FLAGS.generator_type == 'test':
    generator_fn = generator_module.generator_test
  else:
    raise NotImplementedError
  if FLAGS.num_towers != 1 or FLAGS.num_workers != 1:
    raise NotImplementedError(
        'The eval job does not currently support using multiple GPUs')

  # Get activations from real images.
  with tf.device('/device:CPU:1'):
    real_pools, real_images = utils.get_real_activations(
        FLAGS.data_dir,
        local_batch_size,
        FLAGS.eval_sample_size // local_batch_size,
        label_offset=-1,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size)

  num_classes = FLAGS.num_classes
  gen_class_logits = tf.zeros((local_batch_size, num_classes))
  gen_class_ints = tf.multinomial(gen_class_logits, 1)
  gen_sparse_class = tf.squeeze(gen_class_ints)



    # Generate the first batch of generated images and extract activations;
    # this bootstraps the while_loop with a pools and logits tensor.


  test_zs = utils.make_z_normal(1, local_batch_size, FLAGS.z_dim)
  generator = generator_fn(
      test_zs[0],
      gen_sparse_class,
      FLAGS.gf_dim,
      FLAGS.num_classes,
      is_training=False)



  pools, logits = utils.run_custom_inception(
      generator, output_tensor=['pool_3:0', 'logits:0'], graph_def=graph_def)

  # Set up while_loop to compute activations of generated images from generator.
  def while_cond(g_pools, g_logits, i):  # pylint: disable=unused-argument
    return tf.less(i, FLAGS.eval_sample_size // local_batch_size)

  # We use a while loop because we want to generate a batch of images
  # and then feed that batch through Inception to retrieve the activations.
  # Otherwise, if we generate all the samples first and then compute all the
  # activations, we will run out of memory.
  def while_body(g_pools, g_logits, i):
    with tf.control_dependencies([g_pools, g_logits]):

      test_zs = utils.make_z_normal(1, local_batch_size, FLAGS.z_dim)
      # Uniform distribution
      gen_class_logits = tf.zeros((local_batch_size, num_classes))
      gen_class_ints = tf.multinomial(gen_class_logits, 1)
      gen_sparse_class = tf.squeeze(gen_class_ints)

      generator = generator_fn(
        test_zs[0],
        gen_sparse_class,
        FLAGS.gf_dim,
        FLAGS.num_classes,
        is_training=False)

      pools, logits = utils.run_custom_inception(
          generator,
          output_tensor=['pool_3:0', 'logits:0'],
          graph_def=graph_def)
      g_pools = tf.concat([g_pools, pools], 0)
      g_logits = tf.concat([g_logits, logits], 0)

      return (g_pools, g_logits, tf.add(i, 1))

  # Get the activations
  i = tf.constant(1)
  new_generator_pools_list, new_generator_logits_list, _ = tf.while_loop(
      while_cond,
      while_body, [pools, logits, i],
      shape_invariants=[
          tf.TensorShape([None, 2048]),
          tf.TensorShape([None, 1008]),
          i.get_shape()
      ],
      parallel_iterations=1,
      back_prop=False,
      swap_memory=True,
      name='GeneratedActivations')

  new_generator_pools_list.set_shape([FLAGS.eval_sample_size, 2048])
  new_generator_logits_list.set_shape([FLAGS.eval_sample_size, 1008])

  # Get a small batch of samples from generator to dispaly in TensorBoard
  vis_batch_size = 16
  eval_vis_zs = utils.make_z_normal(
      1, vis_batch_size, FLAGS.z_dim)

  gen_class_logits_vis = tf.zeros((vis_batch_size, num_classes))
  gen_class_ints_vis = tf.multinomial(gen_class_logits_vis, 1)
  gen_sparse_class_vis = tf.squeeze(gen_class_ints_vis)

  eval_vis_images = generator_fn(
      eval_vis_zs[0],
      gen_sparse_class_vis,
      FLAGS.gf_dim,
      FLAGS.num_classes,
      is_training=False
      )
  eval_vis_images = tf.cast((eval_vis_images + 1.) * 127.5, tf.uint8)

  with tf.variable_scope('eval_vis'):
    tf.summary.image('generated_images', eval_vis_images)
    tf.summary.image('real_images', real_images)
    tf.summary.image('real_images_grid',
                     tfgan.eval.image_grid(
                         real_images[:16],
                         grid_shape=utils.squarest_grid_size(16),
                         image_shape=(128, 128)))
    tf.summary.image('generated_images_grid',
                     tfgan.eval.image_grid(
                         eval_vis_images[:16],
                         grid_shape=utils.squarest_grid_size(16),
                         image_shape=(128, 128)))

  # Use the activations from the real images and generated images to compute
  # Inception score and FID.
  generated_logits = tf.concat(new_generator_logits_list, 0)
  generated_pools = tf.concat(new_generator_pools_list, 0)

  # Compute Frechet Inception Distance and Inception score
  incscore = tfgan.eval.classifier_score_from_logits(generated_logits)
  fid = tfgan.eval.frechet_classifier_distance_from_activations(
      real_pools, generated_pools)

  with tf.variable_scope('eval'):
    tf.summary.scalar('fid', fid)
    tf.summary.scalar('incscore', incscore)

  session_config = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False)

  tf.contrib.training.evaluate_repeatedly(
      checkpoint_dir=checkpoint_dir,
      hooks=[
          tf.contrib.training.SummaryAtEndHook(log_dir),
          tf.contrib.training.StopAfterNEvalsHook(1)
      ],
      config=session_config)


if __name__ == '__main__':
  tf.app.run()
