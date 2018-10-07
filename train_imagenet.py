"""Train Imagenet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import tensorflow as tf

import utils_ori as utils
import model



tfgan = tf.contrib.gan
gfile = tf.gfile


flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use. [local]')

# flags.DEFINE_string('checkpoint_dir', '/usr/local/google/home/zhanghan/Documents/Research/model_output',
                    # 'Directory name to save the checkpoints. [checkpoint]')
flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                    'Directory name to save the checkpoints. [checkpoint]')
flags.DEFINE_integer('batch_size', 64, 'Number of images in input batch. [64]') # ori 16
flags.DEFINE_integer('shuffle_buffer_size', 100000, 'Number of records to load '
                     'before shuffling and yielding for consumption. [100000]')
flags.DEFINE_integer('save_summaries_steps', 200, 'Number of seconds between '
                     'saving summary statistics. [1]')  # default 300
flags.DEFINE_integer('save_checkpoint_secs', 1200, 'Number of seconds between '
                     'saving checkpoints of model. [1200]')
flags.DEFINE_boolean('is_train', True, 'True for training. [default: True]')
flags.DEFINE_boolean('is_gd_equal', True, 'True for 1:1, False for 1:5')

# TODO(olganw) Find the best way to clean up these flags for eval and train.
flags.DEFINE_integer('task', 0, 'The task id of the current worker. [0]')
flags.DEFINE_integer('ps_tasks', 0, 'The number of ps tasks. [0]')
flags.DEFINE_integer('num_workers', 1, 'The number of worker tasks. [1]')
flags.DEFINE_integer('replicas_to_aggregate', 1, 'The number of replicas '
                     'to aggregate for synchronous optimization [1]')
flags.DEFINE_boolean('sync_replicas', True, 'Whether to sync replicas. [True]')
flags.DEFINE_integer('num_towers', 4, 'The number of GPUs to use per task. [1]')
flags.DEFINE_integer('d_step', 1, 'The number of D_step')
flags.DEFINE_integer('g_step', 1, 'The number of G_step')

# flags.DEFINE_integer('z_dim', 128, 'The dimension of z')

FLAGS = flags.FLAGS



def main(_, is_test=False):
  print('d_learning_rate', FLAGS.discriminator_learning_rate)
  print('g_learning_rate', FLAGS.generator_learning_rate)
  print('data_dir', FLAGS.data_dir)
  print(FLAGS.loss_type, FLAGS.batch_size, FLAGS.beta1)
  print('gf_df_dim', FLAGS.gf_dim, FLAGS.df_dim)
  print('Starting the program..')
  gfile.MakeDirs(FLAGS.checkpoint_dir)

  model_dir = '%s_%s' % ('imagenet', FLAGS.batch_size)
  logdir = os.path.join(FLAGS.checkpoint_dir, model_dir)
  gfile.MakeDirs(logdir)

  graph = tf.Graph()
  with graph.as_default():

    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      # Instantiate global_step.
      global_step = tf.train.create_global_step()

    # Create model with FLAGS, global_step, and devices.
    devices = ['/gpu:{}'.format(tower) for tower in range(FLAGS.num_towers)]

    # Create noise tensors
    zs = utils.make_z_normal(
        FLAGS.num_towers, FLAGS.batch_size, FLAGS.z_dim)

    print('save_summaries_steps', FLAGS.save_summaries_steps)

    dcgan = model.SNGAN(
        zs=zs,
        config=FLAGS,
        global_step=global_step,
        devices=devices)

    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      # Create sync_hooks when needed.
      if FLAGS.sync_replicas and FLAGS.num_workers > 1:
        print('condition 1')
        sync_hooks = [
            dcgan.d_opt.make_session_run_hook(FLAGS.task == 0),
            dcgan.g_opt.make_session_run_hook(FLAGS.task == 0)
        ]
      else:
        print('condition 2')
        sync_hooks = []

    train_ops = tfgan.GANTrainOps(
        generator_train_op=dcgan.g_optim,
        discriminator_train_op=dcgan.d_optim,
        global_step_inc_op=dcgan.increment_global_step)


    # We set allow_soft_placement to be True because Saver for the DCGAN model
    # gets misplaced on the GPU.
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)

    if is_test:
      return graph

    print("G step: ", FLAGS.g_step)
    print("D_step: ", FLAGS.d_step)
    train_steps = tfgan.GANTrainSteps(FLAGS.g_step, FLAGS.d_step)

    tfgan.gan_train(
        train_ops,
        get_hooks_fn=tfgan.get_sequential_train_hooks(
            train_steps=train_steps),
        hooks=([tf.train.StopAtStepHook(num_steps=2000000)] + sync_hooks),
        logdir=logdir,
        # master=FLAGS.master,
        # scaffold=scaffold, # load from google checkpoint
        is_chief=(FLAGS.task == 0),
        save_summaries_steps=FLAGS.save_summaries_steps,
        save_checkpoint_secs=FLAGS.save_checkpoint_secs,
        config=session_config)


if __name__ == '__main__':
  tf.app.run()
