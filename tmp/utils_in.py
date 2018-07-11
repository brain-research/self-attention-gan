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

import math
import os
import sympy
import numpy as np
import tensorflow as tf


def make_z_normal(num_batches, batch_size, z_dim):
  shape = [num_batches, batch_size, z_dim]
  z = tf.random_normal(shape, name='z0', dtype=tf.float32)
  return z


def get_imagenet_batches(data_dir,
                         batch_size,
                         num_batches,
                         label_offset=0,
                         cycle_length=1,
                         shuffle_buffer_size=100000):
  filenames = tf.gfile.Glob(os.path.join(data_dir, '*_train_*-*-of-*'))
  filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
  filename_dataset = filename_dataset.shuffle(len(filenames))
  prefetch = max(int((batch_size * num_batches) / cycle_length), 1)
  dataset = filename_dataset.interleave(
      lambda fn: tf.data.RecordIODataset(fn).prefetch(prefetch),
      cycle_length=cycle_length)

  dataset = dataset.shuffle(shuffle_buffer_size)
  image_size = 128

  def _extract_image_and_label(record):
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
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  dataset = dataset.batch(num_batches)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  batches = []
  for i in range(num_batches):
    im = images[i, ...]
    im.set_shape([batch_size, image_size, image_size, 3])

    lb = labels[i, ...]
    lb.set_shape((batch_size,))

    batches.append((im, tf.expand_dims(lb, 1)))

  return batches


def squarest_grid_size(num_images):
  divisors = sympy.divisors(num_images)
  square_root = math.sqrt(num_images)
  width = 1
  for d in divisors:
    if d > square_root:
      break
    width = d
  return (num_images // width, width)
