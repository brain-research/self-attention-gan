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
import os
IMAGE_SIZE=128

def _extract_image_and_label(record):
  """Extracts and preprocesses the image and label from the record."""
  features = tf.parse_single_example(
    record,
    features={
      'image_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64)
    })
  image_size = IMAGE_SIZE
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape(image_size * image_size * 3)
  image = tf.reshape(image, [image_size, image_size, 3])

  image = tf.cast(image, tf.float32) * (2. / 255) - 1.

  label = tf.cast(features['label'], tf.int32)

  return image, label

class InputFunction(object):
  """Wrapper class that is passed as callable to Estimator."""

  def __init__(self, is_training, noise_dim, dataset_name, num_classes, data_dir="./dataset",
               cycle_length=64, shuffle_buffer_size=100000):
    self.is_training = is_training
    self.noise_dim = noise_dim
    split = ('train' if is_training else 'test')
    self.data_files = tf.gfile.Glob(os.path.join(data_dir, '*.tfrecords'))
    self.parser = _extract_image_and_label
    self.num_classes = num_classes
    self.cycle_length = cycle_length
    self.shuffle_buffer_size = shuffle_buffer_size

  def __call__(self, params):
    """Creates a simple Dataset pipeline."""

    batch_size = params['batch_size']
    filename_dataset = tf.data.Dataset.from_tensor_slices(self.data_files)
    filename_dataset = filename_dataset.shuffle(len(self.data_files))

    def tfrecord_dataset(filename):
      buffer_size = 8 * 1024 * 1224
      return tf.data.TFRecordDataset(filename, buffer_size=buffer_size)

    dataset = filename_dataset.apply(tf.contrib.data.parallel_interleave(
        tfrecord_dataset,
        cycle_length=self.cycle_length, sloppy=True))
    if self.is_training:
      dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
          self.shuffle_buffer_size, -1))
    dataset = dataset.map(self.parser, num_parallel_calls=32)
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(4)    # Prefetch overlaps in-feed with training
    images, labels = dataset.make_one_shot_iterator().get_next()
    labels = tf.squeeze(labels)
    random_noise = tf.random_normal([batch_size, self.noise_dim])

    gen_class_logits = tf.zeros((batch_size, self.num_classes))
    gen_class_ints = tf.multinomial(gen_class_logits, 1)
    gen_sparse_class = tf.squeeze(gen_class_ints)

    features = {
        'real_images': images,
        'random_noise': random_noise,
        'fake_labels': gen_sparse_class}

    return features, labels

