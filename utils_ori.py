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

"""Some codes from https://github.com/Newmu/dcgan_code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import numpy as np
import scipy.misc
import sympy
import tensorflow as tf

tfgan = tf.contrib.gan
classifier_metrics = tf.contrib.gan.eval.classifier_metrics
gfile = tf.gfile




def make_z_normal(num_batches, batch_size, z_dim):
  """Make random noises tensors with normal distribution  feeding into the generator
  Args:
    num_batches: copies of batches
    batch_size: the batch_size for z
    z_dim: The dimension of the z (noise) vector.
  Returns:
    zs:  noise tensors.
  """
  shape = [num_batches, batch_size, z_dim]
  z = tf.random_normal(shape,  name='z0', dtype=tf.float32)
  return z


def run_custom_inception(
    images,
    output_tensor,
    graph_def=None,
    # image_size=classifier_metrics.INCEPTION_DEFAULT_IMAGE_SIZE):
    image_size=299):
    # input_tensor=classifier_metrics.INCEPTION_V1_INPUT):
  """Run images through a pretrained Inception classifier.

  This method resizes the images before feeding them through Inception; we do
  this to accommodate feeding images through in minibatches without having to
  construct any large tensors.

  Args:
    images: Input tensors. Must be [batch, height, width, channels]. Input shape
      and values must be in [-1, 1], which can be achieved using
      `preprocess_image`.
    output_tensor: Name of output Tensor. This function will compute activations
      at the specified layer. Examples include INCEPTION_V3_OUTPUT and
      INCEPTION_V3_FINAL_POOL which would result in this function computing
      the final logits or the penultimate pooling layer.
    graph_def: A GraphDef proto of a pretrained Inception graph. If `None`,
      call `default_graph_def_fn` to get GraphDef.
    image_size: Required image width and height. See unit tests for the default
      values.
    input_tensor: Name of input Tensor.

  Returns:
    Logits.
  """

  images = tf.image.resize_bilinear(images, [image_size, image_size])

  return tfgan.eval.run_inception(
      images,
      graph_def=graph_def,
      image_size=image_size,
      # input_tensor=input_tensor,
      output_tensor=output_tensor)


def get_real_activations(data_dir,
                         batch_size,
                         num_batches,
                         label_offset=0,
                         cycle_length=1,
                         shuffle_buffer_size=100000):
  """Fetches num_batches batches of size batch_size from the data_dir.

  Args:
    data_dir: The directory to read data from. Expected to be a single
        TFRecords file.
    batch_size: The number of elements in a single minibatch.
    num_batches: The number of batches to fetch at a time.
    label_offset: The scalar to add to the labels in the dataset. The imagenet
        GAN code expects labels in [0, 999], and this scalar can be used to move
        other labels into this range. (Default: 0)
    cycle_length: The number of input elements to process concurrently in the
        Dataset loader. (Default: 1)
    shuffle_buffer_size: The number of records to load before shuffling. Larger
        means more likely randomization. (Default: 100000)
  Returns:
    A list of num_batches batches of size batch_size.
  """
  # filenames = gfile.Glob(os.path.join(data_dir, '*_train_*-*-of-*'))

  filenames = tf.gfile.Glob(os.path.join(data_dir, '*.tfrecords'))
  filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
  filename_dataset = filename_dataset.shuffle(len(filenames))
  prefetch = max(int((batch_size * num_batches) / cycle_length), 1)
  dataset = filename_dataset.interleave(
      lambda fn: tf.data.TFRecordDataset(fn).prefetch(prefetch),
      cycle_length=cycle_length)

  dataset = dataset.shuffle(shuffle_buffer_size)
  image_size = 128
  # graph_def = classifier_metrics._default_graph_def_fn()  # pylint: disable=protected-access

  def _extract_image_and_label(record):
    """Extracts and preprocesses the image and label from the record."""
    features = tf.parse_single_example(
        record,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape(image_size * image_size * 3)
    image = tf.reshape(image, [image_size, image_size, 3])

    image = tf.cast(image, tf.float32) * (2. / 255) - 1.

    label = tf.cast(features['label'], tf.int32)
    label += label_offset

    return image, label

  dataset = dataset.map(
      _extract_image_and_label,
      num_parallel_calls=16).prefetch(batch_size * num_batches)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()

  real_images, _ = iterator.get_next()
  real_images.set_shape([batch_size, image_size, image_size, 3])

  pools = run_custom_inception(
      real_images, graph_def=None, output_tensor=['pool_3:0'])[0]

  def while_cond(_, i):
    return tf.less(i, num_batches)

  def while_body(real_pools, i):
    with tf.control_dependencies([real_pools]):
      imgs, _ = iterator.get_next()
      imgs.set_shape([batch_size, image_size, image_size, 3])
      pools = run_custom_inception(
          imgs, graph_def=None, output_tensor=['pool_3:0'])[0]
      real_pools = tf.concat([real_pools, pools], 0)
      return (real_pools, tf.add(i, 1))

  # Get activations from real images.
  i = tf.constant(1)
  real_pools, _ = tf.while_loop(
      while_cond,
      while_body, [pools, i],
      shape_invariants=[tf.TensorShape([None, 2048]),
                        i.get_shape()],
      parallel_iterations=1,
      back_prop=False,
      swap_memory=True,
      name='RealActivations')

  real_pools.set_shape([batch_size * num_batches, 2048])

  return real_pools, real_images


def get_imagenet_batches(data_dir,
                         batch_size,
                         num_batches,
                         label_offset=0,
                         cycle_length=1,
                         shuffle_buffer_size=100000):
  """Fetches num_batches batches of size batch_size from the data_dir.

  Args:
    data_dir: The directory to read data from. Expected to be a single
        TFRecords file.
    batch_size: The number of elements in a single minibatch.
    num_batches: The number of batches to fetch at a time.
    label_offset: The scalar to add to the labels in the dataset. The imagenet
        GAN code expects labels in [0, 999], and this scalar can be used to move
        other labels into this range. (Default: 0)
    cycle_length: The number of input elements to process concurrently in the
        Dataset loader. (Default: 1)
    shuffle_buffer_size: The number of records to load before shuffling. Larger
        means more likely randomization. (Default: 100000)
  Returns:
    A list of num_batches batches of size batch_size.
  """
  # filenames = gfile.Glob(os.path.join(data_dir, '*_train_*-*-of-*'))
  filenames = tf.gfile.Glob(os.path.join(data_dir, '*.tfrecords'))
  filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
  filename_dataset = filename_dataset.shuffle(len(filenames))
  prefetch = max(int((batch_size * num_batches) / cycle_length), 1)
  dataset = filename_dataset.interleave(
      lambda fn: tf.data.TFRecordDataset(fn).prefetch(prefetch),
      cycle_length=cycle_length)

  dataset = dataset.shuffle(shuffle_buffer_size)
  image_size = 128

  def _extract_image_and_label(record):
    """Extracts and preprocesses the image and label from the record."""
    features = tf.parse_single_example(
        record,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape(image_size * image_size * 3)
    image = tf.reshape(image, [image_size, image_size, 3])

    image = tf.cast(image, tf.float32) * (2. / 255) - 1.

    label = tf.cast(features['label'], tf.int32)
    label += label_offset

    return image, label

  dataset = dataset.map(
      _extract_image_and_label,
      num_parallel_calls=16).prefetch(batch_size * num_batches)
  dataset = dataset.repeat()  # Repeat for unlimited epochs.
  dataset = dataset.batch(batch_size)
  dataset = dataset.batch(num_batches)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  batches = []
  for i in range(num_batches):
    # Dataset batches lose shape information. Put it back in.
    im = images[i, ...]
    im.set_shape([batch_size, image_size, image_size, 3])

    lb = labels[i, ...]
    lb.set_shape((batch_size,))

    batches.append((im, tf.expand_dims(lb, 1)))

  return batches


def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))

  idx = 0
  for i in range(0, size[0]):
    for j in range(0, size[1]):
      img[j * h:(j + 1) * h, i * w:(i + 1) * w, :] = images[idx]
      idx += 1
  return img


def imsave(images, size, path):
  with gfile.Open(path, mode='w') as f:
    saved = scipy.misc.imsave(f, merge(images, size))
  return saved


def inverse_transform(images):
  return (images + 1.) / 2.


def visualize(sess, dcgan, config, option):
  option = 0
  if option == 0:
    all_samples = []
    for i in range(484):
      print(i)
      samples = sess.run(dcgan.generator)
      all_samples.append(samples)
    samples = np.concatenate(all_samples, 0)
    n = int(np.sqrt(samples.shape[0]))
    m = samples.shape[0] // n
    save_images(samples, [m, n], './' + config.sample_dir + '/test.png')
  elif option == 1:
    counter = 0
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)
    while counter < 1005:
      print(counter)
      samples, fake = sess.run([dcgan.generator, dcgan.d_loss_class])
      fake = np.argsort(fake)
      print(np.sum(samples))
      print(fake)
      for i in range(samples.shape[0]):
        name = '%s%d.png' % (chr(ord('a') + counter % 10), counter)
        img = np.expand_dims(samples[fake[i]], 0)
        if counter >= 1000:
          save_images(img, [1, 1], './{}/turk/fake{}.png'.format(
              config.sample_dir, counter - 1000))
        else:
          save_images(img, [1, 1], './{}/turk/{}'.format(
              config.sample_dir, name))
        counter += 1
  elif option == 2:
    values = np.arange(0, 1, 1. / config.batch_size)
    for idx in range(100):
      print(' [*] %d' % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      save_images(samples, [8, 8], './{}/test_arange_{}.png'.format(
          config.sample_dir, idx))


def squarest_grid_size(num_images):
  """Calculates the size of the most square grid for num_images.

  Calculates the largest integer divisor of num_images less than or equal to
  sqrt(num_images) and returns that as the width. The height is
  num_images / width.

  Args:
    num_images: The total number of images.

  Returns:
    A tuple of (height, width) for the image grid.
  """
  divisors = sympy.divisors(num_images)
  square_root = math.sqrt(num_images)
  width = 1
  for d in divisors:
    if d > square_root:
      break
    width = d
  return (num_images // width, width)
