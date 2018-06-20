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
"""Utility functions used in the sngan."""
import numpy as np
import tensorflow as tf
import math
import os
import numpy as np
import scipy.misc
import sympy
import tensorflow as tf

tfgan = tf.contrib.gan




def get_inception_scores(images, batch_size, num_inception_images):
  """Gets Inception score for some images.

  Args:
    images: Image minibatch. Shape [batch size, width, height, channels]. Values
      are in [-1, 1].
    batch_size: Python integer. Batch dimension.
    num_inception_images: Number of images to run through Inception at once.

  Returns:
    Inception scores. Tensor shape is [batch size].

  Raises:
    ValueError: If `batch_size` is incompatible with the first dimension of
      `images`.
    ValueError: If `batch_size` isn't divisible by `num_inception_images`.
  """
  # Validate inputs.
  images.shape[0:1].assert_is_compatible_with([batch_size])
  if batch_size % num_inception_images != 0:
    raise ValueError(
        "`batch_size` must be divisible by `num_inception_images`.")

  # Resize images.
  size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
  resized_images = tf.image.resize_bilinear(images, [size, size])

  # Run images through Inception.
  num_batches = batch_size // num_inception_images
  inc_score = tfgan.eval.inception_score(
      resized_images, num_batches=num_batches)

  return inc_score


def get_frechet_inception_distance(real_images, generated_images, batch_size,
                                   num_inception_images):
  """Gets Frechet Inception Distance between real and generated images.

  Args:
    real_images: Real images minibatch. Shape [batch size, width, height,
      channels. Values are in [-1, 1].
    generated_images: Generated images minibatch. Shape [batch size, width,
      height, channels]. Values are in [-1, 1].
    batch_size: Python integer. Batch dimension.
    num_inception_images: Number of images to run through Inception at once.

  Returns:
    Frechet Inception distance. A floating-point scalar.

  Raises:
    ValueError: If the minibatch size is known at graph construction time, and
      doesn't batch `batch_size`.
  """
  # Validate input dimensions.
  real_images.shape[0:1].assert_is_compatible_with([batch_size])
  generated_images.shape[0:1].assert_is_compatible_with([batch_size])

  # Resize input images.
  size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
  resized_real_images = tf.image.resize_bilinear(real_images, [size, size])
  resized_generated_images = tf.image.resize_bilinear(
      generated_images, [size, size])

  # Compute Frechet Inception Distance.
  num_batches = batch_size // num_inception_images
  fid = tfgan.eval.frechet_inception_distance(
      resized_real_images, resized_generated_images, num_batches=num_batches)

  return fid


def simple_summary(summary_writer, summary_dict, step):
  """Takes a dictionary w/ scalar values. Prints it and logs to tensorboard.

  Args:
    summary_writer: The tensorflow summary writer.
    summary_dict: The dictionary contains all the key-value pairs of data.
    step: The current logging step.
  """
  print(("Step: %s\t" % step) + "\t".join(
      ["%s: %s" % (k, v) for k, v in summary_dict.items()]
  ))
  summary = tf.Summary(value=[
      tf.Summary.Value(tag=k, simple_value=v) for k, v in summary_dict.items()
  ])
  summary_writer.add_summary(summary, global_step=step)


def calculate_inception_score(images, batch_size, num_inception_images):
  """The function wraps the inception score calculation.

  Args:
    images: The numpy array of images, pixel value is from [-1, 1].
    batch_size: The total number of images for inception score calculation.
    num_inception_images: Number of images to run through Inception at once.
  Returns:
    Inception Score. A floating-point scalar.
  """
  with tf.Graph().as_default(), tf.Session() as sess:
    images_op = tf.convert_to_tensor(
        np.array(images)
    )
    score_op = get_inception_scores(images_op, batch_size, num_inception_images)
    return sess.run(score_op)


def calculate_fid(real_images, generated_images, batch_size,
                  num_inception_images):
  """The function wraps to the FID calculation.

  Args:
    real_images: The numpy array of real images, pixel value is from [-1, 1].
    generated_images: The numpy array of generated images, pixel value is
                      from [-1, 1].
    batch_size: The total number of images for inception score calculation
    num_inception_images: Number of images to run through Inception at once.
  Returns:
    Frechet Inception distance. A floating-point scalar.
  """
  with tf.Graph().as_default(), tf.Session() as sess:
    real_images_op = tf.convert_to_tensor(
        np.array(real_images)
    )
    generated_images_op = tf.convert_to_tensor(
        np.array(generated_images)
    )
    fid = get_frechet_inception_distance(real_images_op, generated_images_op,
                                         batch_size, num_inception_images)
    return sess.run(fid)
