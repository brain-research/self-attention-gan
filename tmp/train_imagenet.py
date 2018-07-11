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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
import tensorflow as tf
import model
import utils_in as utils

tfgan = tf.contrib.gan


flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use. [local]')
flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                    'Directory name to save the checkpoints. [checkpoint]')
flags.DEFINE_integer('batch_size', 32, 'Number of images in input batch. [64]')
flags.DEFINE_integer('shuffle_buffer_size', 100000, 'Number of records to load '
                     'before shuffling and yielding for consumption. [100000]')
flags.DEFINE_integer('save_summaries_steps', 1, 'Number of seconds between '
                     'saving summary statistics. [1]')
flags.DEFINE_integer('save_checkpoint_secs', 1200, 'Number of seconds between '
                     'saving checkpoints of model. [1200]')
flags.DEFINE_integer('task', 0, 'The task id of the current worker. [0]')
flags.DEFINE_integer('ps_tasks', 0, 'The number of ps tasks. [0]')
flags.DEFINE_integer('num_workers', 1, 'The number of worker tasks. [1]')
flags.DEFINE_integer('replicas_to_aggregate', 1, 'The number of replicas '
                     'to aggregate for synchronous optimization [1]')
flags.DEFINE_boolean('sync_replicas', True, 'Whether to sync replicas. [True]')
flags.DEFINE_integer('num_towers', 1, 'The number of GPUs to use per task. [1]')

FLAGS = flags.FLAGS


def main(_):
  tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
  model_dir = '%s_%s' % ('imagenet', FLAGS.batch_size)
  logdir = os.path.join(FLAGS.checkpoint_dir, model_dir)
  tf.gfile.MakeDirs(logdir)
  graph = tf.Graph()
  with graph.as_default():
    global_step = tf.train.create_global_step()
    devices = ['/gpu:{}'.format(tower) for tower in range(FLAGS.num_towers)]

    noise_tensor = utils.make_z_normal(
        FLAGS.num_towers, FLAGS.batch_size, FLAGS.z_dim)

    model_object = model.SNGAN(
        noise_tensor=noise_tensor,
        config=FLAGS,
        global_step=global_step,
        devices=devices)

    train_ops = tfgan.GANTrainOps(
        generator_train_op=model_object.g_optim,
        discriminator_train_op=model_object.d_optim,
        global_step_inc_op=model_object.increment_global_step)

    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)

    train_steps = tfgan.GANTrainSteps(1, 1)
    tfgan.gan_train(
        train_ops,
        get_hooks_fn=tfgan.get_sequential_train_hooks(
            train_steps=train_steps),
        hooks=([tf.train.StopAtStepHook(num_steps=2000000)]),
        logdir=logdir,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        save_summaries_steps=FLAGS.save_summaries_steps,
        save_checkpoint_secs=FLAGS.save_checkpoint_secs,
        config=session_config)

if __name__ == '__main__':
  tf.app.run()
