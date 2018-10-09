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

"""The DCGAN Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

import discriminator as disc
import generator as generator_module

import ops
import utils_ori as utils


tfgan = tf.contrib.gan

flags.DEFINE_string(
    # 'data_dir', '/gpu/hz138/Data/imagenet', #'/home/hz138/Data/imagenet',
    'data_dir', '/bigdata1/hz138/Data/imagenet',
    'Directory with Imagenet input data as sharded recordio files of pre-'
    'processed images.')
flags.DEFINE_float('discriminator_learning_rate', 0.0004,
                   'Learning rate of for adam. [0.0004]')
flags.DEFINE_float('generator_learning_rate', 0.0001,
                   'Learning rate of for adam. [0.0004]')
flags.DEFINE_float('beta1', 0.0, 'Momentum term of adam. [0.5]')
flags.DEFINE_integer('image_size', 128, 'The size of image to use '
                     '(will be center cropped) [128]')
flags.DEFINE_integer('image_width', 128,
                     'The width of the images presented to the model')
flags.DEFINE_integer('data_parallelism', 64, 'The number of objects to read at'
                     ' one time when loading input data. [64]')
flags.DEFINE_integer('z_dim', 128, 'Dimensionality of latent code z. [8192]')
flags.DEFINE_integer('gf_dim', 64, 'Dimensionality of gf. [64]')
flags.DEFINE_integer('df_dim', 64, 'Dimensionality of df. [64]')
flags.DEFINE_integer('number_classes', 1000, 'The number of classes in the dataset')
flags.DEFINE_string('loss_type', 'hinge_loss', 'the loss type can be'
                    ' hinge_loss or kl_loss')
flags.DEFINE_string('generator_type', 'test', 'test or baseline')
flags.DEFINE_string('discriminator_type', 'test', 'test or baseline')



FLAGS = flags.FLAGS


def _get_d_real_loss(discriminator_on_data_logits):
  loss = tf.nn.relu(1.0 - discriminator_on_data_logits)
  return tf.reduce_mean(loss)


def _get_d_fake_loss(discriminator_on_generator_logits):
  return tf.reduce_mean(tf.nn.relu(1 + discriminator_on_generator_logits))


def _get_g_loss(discriminator_on_generator_logits):
  return -tf.reduce_mean(discriminator_on_generator_logits)


def _get_d_real_loss_KL(discriminator_on_data_logits):
  loss = tf.nn.softplus(-discriminator_on_data_logits)
  return tf.reduce_mean(loss)


def _get_d_fake_loss_KL(discriminator_on_generator_logits):
  return tf.reduce_mean(tf.nn.softplus(discriminator_on_generator_logits))


def _get_g_loss_KL(discriminator_on_generator_logits):
  return tf.reduce_mean(-discriminator_on_generator_logits)



class SNGAN(object):
  """SNGAN model."""

  def __init__(self, zs, config=None, global_step=None, devices=None):
    """Initializes the DCGAN model.

    Args:
      zs: input noise tensors for the generator
      config: the configuration FLAGS object
      global_step: the global training step (maintained by the supervisor)
      devices: the list of device names to place ops on (multitower training)
    """

    self.config = config
    self.image_size = FLAGS.image_size
    self.image_shape = [FLAGS.image_size, FLAGS.image_size, 3]
    self.z_dim = FLAGS.z_dim
    self.gf_dim = FLAGS.gf_dim
    self.df_dim = FLAGS.df_dim
    self.num_classes = FLAGS.number_classes

    self.data_parallelism = FLAGS.data_parallelism
    self.zs = zs

    self.c_dim = 3
    self.dataset_name = 'imagenet'
    self.devices = devices
    self.global_step = global_step

    self.build_model()

  def build_model(self):
    """Builds a model."""
    config = self.config
    # If ps_tasks is zero, the local device is used. When using multiple
    # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
    # across the different devices.
    current_step = tf.cast(self.global_step, tf.float32)
    # g_ratio = (1.0 + 2e-5 * tf.maximum((current_step - 100000.0), 0.0))
    # g_ratio = tf.minimum(g_ratio, 4.0)
    self.d_learning_rate = FLAGS.discriminator_learning_rate
    self.g_learning_rate = FLAGS.generator_learning_rate
    # self.g_learning_rate = FLAGS.generator_learning_rate / (1.0 + 2e-5 * tf.cast(self.global_step, tf.float32))
    # self.g_learning_rate = FLAGS.generator_learning_rate / g_ratio
    with tf.device(tf.train.replica_device_setter(config.ps_tasks)):
      self.d_opt = tf.train.AdamOptimizer(
          self.d_learning_rate, beta1=FLAGS.beta1)
      self.g_opt = tf.train.AdamOptimizer(
          self.g_learning_rate, beta1=FLAGS.beta1)
      if config.sync_replicas and config.num_workers > 1:
        self.d_opt = tf.train.SyncReplicasOptimizer(
            opt=self.d_opt, replicas_to_aggregate=config.replicas_to_aggregate)
        self.g_opt = tf.train.SyncReplicasOptimizer(
            opt=self.g_opt, replicas_to_aggregate=config.replicas_to_aggregate)

    if config.num_towers > 1:
      all_d_grads = []
      all_g_grads = []
      for idx, device in enumerate(self.devices):
        with tf.device('/%s' % device):
          with tf.name_scope('device_%s' % idx):
            with ops.variables_on_gpu0():
              self.build_model_single_gpu(
                  gpu_idx=idx,
                  batch_size=config.batch_size,
                  num_towers=config.num_towers)
              d_grads = self.d_opt.compute_gradients(self.d_losses[-1],
                                                     var_list=self.d_vars)
              g_grads = self.g_opt.compute_gradients(self.g_losses[-1],
                                                     var_list=self.g_vars)
              all_d_grads.append(d_grads)
              all_g_grads.append(g_grads)
      d_grads = ops.avg_grads(all_d_grads)
      g_grads = ops.avg_grads(all_g_grads)
    else:
      with tf.device(tf.train.replica_device_setter(config.ps_tasks)):
        # TODO(olganw): reusing virtual batchnorm doesn't work in the multi-
        # replica case.
        self.build_model_single_gpu(batch_size=config.batch_size,
                                    num_towers=config.num_towers)
        d_grads = self.d_opt.compute_gradients(self.d_losses[-1],
                                               var_list=self.d_vars)
        g_grads = self.g_opt.compute_gradients(self.g_losses[-1],
                                               var_list=self.g_vars)
    with tf.device(tf.train.replica_device_setter(config.ps_tasks)):
      update_moving_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      print('update_moving_ops', update_moving_ops)
      if config.sync_replicas:
        with tf.control_dependencies(update_moving_ops):
          d_step = tf.get_variable('d_step', initializer=0, trainable=False)
          self.d_optim = self.d_opt.apply_gradients(d_grads, global_step=d_step)
          g_step = tf.get_variable('g_step', initializer=0, trainable=False)
          self.g_optim = self.g_opt.apply_gradients(g_grads, global_step=g_step)
      else:
        # Don't create any additional counters, and don't update the global step
        with tf.control_dependencies(update_moving_ops):
          self.d_optim = self.d_opt.apply_gradients(d_grads)
          self.g_optim = self.g_opt.apply_gradients(g_grads)

  def build_model_single_gpu(self, gpu_idx=0, batch_size=1, num_towers=1):
    """Builds a model for a single GPU.

    Args:
      gpu_idx: The index of the gpu in the tower.
      batch_size: The minibatch size. (Default: 1)
      num_towers: The total number of towers in this model. (Default: 1)
    """
    config = self.config
    show_num = min(config.batch_size, 64)

    reuse_vars = gpu_idx > 0
    if gpu_idx == 0:
      self.increment_global_step = self.global_step.assign_add(1)
      self.batches = utils.get_imagenet_batches(
          FLAGS.data_dir, batch_size, num_towers, label_offset=0,
          cycle_length=config.data_parallelism,
          shuffle_buffer_size=config.shuffle_buffer_size)
      sample_images, _ = self.batches[0]
      vis_images = tf.cast((sample_images + 1.) * 127.5, tf.uint8)
      tf.summary.image('input_image_grid',
                       tfgan.eval.image_grid(
                           vis_images[:show_num],
                           grid_shape=utils.squarest_grid_size(
                               show_num),
                           image_shape=(128, 128)))

    images, sparse_labels = self.batches[gpu_idx]
    sparse_labels = tf.squeeze(sparse_labels)
    print('han spase_labels.shape', sparse_labels.shape)

    gen_class_logits = tf.zeros((batch_size, self.num_classes))
    gen_class_ints = tf.multinomial(gen_class_logits, 1)
    # gen_sparse_class = tf.argmax(gen_class_ints, axis=1)  BIG BUG!!!!!
    gen_sparse_class = tf.squeeze(gen_class_ints)
    print('han gen_sparse_class.shape', gen_sparse_class.shape)
    assert len(gen_class_ints.get_shape()) == 2
    gen_class_ints = tf.squeeze(gen_class_ints)
    assert len(gen_class_ints.get_shape()) == 1
    gen_class_vector = tf.one_hot(gen_class_ints, self.num_classes)
    assert len(gen_class_vector.get_shape()) == 2
    assert gen_class_vector.dtype == tf.float32

    if FLAGS.generator_type == 'baseline':
      generator_fn = generator_module.generator
    elif FLAGS.generator_type == 'test':
      generator_fn = generator_module.generator_test

    generator = generator_fn(
        self.zs[gpu_idx],
        gen_sparse_class,
        self.gf_dim,
        self.num_classes
        )


    if gpu_idx == 0:
      generator_means = tf.reduce_mean(generator, 0, keep_dims=True)
      generator_vars = tf.reduce_mean(
          tf.squared_difference(generator, generator_means), 0, keep_dims=True)
      generator = tf.Print(
          generator,
          [tf.reduce_mean(generator_means), tf.reduce_mean(generator_vars)],
          'generator mean and average var', first_n=1)
      image_means = tf.reduce_mean(images, 0, keep_dims=True)
      image_vars = tf.reduce_mean(
          tf.squared_difference(images, image_means), 0, keep_dims=True)
      images = tf.Print(
          images, [tf.reduce_mean(image_means), tf.reduce_mean(image_vars)],
          'image mean and average var', first_n=1)
      sparse_labels = tf.Print(sparse_labels, [sparse_labels, sparse_labels.shape], 'sparse_labels', first_n=2)
      gen_sparse_class = tf.Print(gen_sparse_class, [gen_sparse_class, gen_sparse_class.shape], 'gen_sparse_labels', first_n=2)

      self.generators = []

    self.generators.append(generator)

    if FLAGS.discriminator_type == 'baseline':
      discriminator_fn = disc.discriminator
    elif FLAGS.discriminator_type == 'test':
      discriminator_fn = disc.discriminator_test
    else:
      raise NotImplementedError
    discriminator_on_data_logits = discriminator_fn(images, sparse_labels, self.df_dim, self.num_classes,
                                                    update_collection=None)
    discriminator_on_generator_logits = discriminator_fn(generator, gen_sparse_class, self.df_dim, self.num_classes,
                                                         update_collection="NO_OPS")

    vis_generator = tf.cast((generator + 1.) * 127.5, tf.uint8)
    tf.summary.image('generator', vis_generator)

    tf.summary.image('generator_grid',
                     tfgan.eval.image_grid(
                         vis_generator[:show_num],
                         grid_shape=utils.squarest_grid_size(show_num),
                         image_shape=(128, 128)))

    if FLAGS.loss_type == 'hinge_loss':
      d_loss_real = _get_d_real_loss(
          discriminator_on_data_logits)
      d_loss_fake = _get_d_fake_loss(discriminator_on_generator_logits)
      g_loss_gan = _get_g_loss(discriminator_on_generator_logits)
      print('hinge loss is using')
    elif FLAGS.loss_type == 'kl_loss':
      d_loss_real = _get_d_real_loss_KL(
          discriminator_on_data_logits)
      d_loss_fake = _get_d_fake_loss_KL(discriminator_on_generator_logits)
      g_loss_gan = _get_g_loss_KL(discriminator_on_generator_logits)
      print('kl loss is using')
    else:
      raise NotImplementedError


    d_loss = d_loss_real + d_loss_fake
    g_loss = g_loss_gan


    # add logit log
    logit_discriminator_on_data = tf.reduce_mean(discriminator_on_data_logits)
    logit_discriminator_on_generator = tf.reduce_mean(
        discriminator_on_generator_logits)



    # Add summaries.
    tf.summary.scalar('d_loss', d_loss)
    tf.summary.scalar('d_loss_real', d_loss_real)
    tf.summary.scalar('d_loss_fake', d_loss_fake)
    tf.summary.scalar('g_loss', g_loss)
    tf.summary.scalar('logit_real', logit_discriminator_on_data)
    tf.summary.scalar('logit_fake', logit_discriminator_on_generator)
    tf.summary.scalar('d_learning_rate', self.d_learning_rate)
    tf.summary.scalar('g_learning_rate', self.g_learning_rate)



    if gpu_idx == 0:
      self.d_loss_reals = []
      self.d_loss_fakes = []
      self.d_losses = []
      self.g_losses = []
    self.d_loss_reals.append(d_loss_real)
    self.d_loss_fakes.append(d_loss_fake)
    self.d_losses.append(d_loss)
    self.g_losses.append(g_loss)

    if gpu_idx == 0:
      self.get_vars()
      print('gvars', self.g_vars)
      print('dvars', self.d_vars)
      print('sigma_ratio_vars', self.sigma_ratio_vars)
      for var in self.sigma_ratio_vars:
        tf.summary.scalar(var.name, var)

  def get_vars(self):
    """Get variables."""
    t_vars = tf.trainable_variables()
    # TODO(olganw): scoping or collections for this instead of name hack
    self.d_vars = [var for var in t_vars if var.name.startswith('model/d_')]
    self.g_vars = [var for var in t_vars if var.name.startswith('model/g_')]
    self.sigma_ratio_vars = [var for var in t_vars if 'sigma_ratio' in var.name]
    for x in self.d_vars:
      assert x not in self.g_vars
    for x in self.g_vars:
      assert x not in self.d_vars
    for x in t_vars:
      assert x in  self.g_vars or x in self.d_vars, x.name
    self.all_vars = t_vars
