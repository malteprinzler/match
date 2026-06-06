import tensorflow as tf

_PROJECTION_CLIP_VALUE = 1e10


@tf.function
def fix_nan(tensor: tf.Tensor, default_value: float = 1.0) -> tf.Tensor:
  """Fix NaNs in tensor and replace them with the provided default value."""
  return tf.cond(
      tf.reduce_any(tf.math.is_nan(tensor)),
      lambda: tf.where(tf.math.is_nan(tensor), default_value, tensor),
      lambda: tensor,  # Return the original tensor if no NaNs are found
  )


def fix_and_clip(
    tensor: tf.Tensor, min_value: float = -1e10, max_value: float = 1e10
) -> tf.Tensor:
  """Fix NaNs and Inf in tensor and clip all values to the provided range."""

  # Replace -Inf and Inf with min_value and max_value.
  tensor = tf.where(
      tf.math.is_inf(tensor) & tf.math.less(tensor, 0),
      tf.fill(tf.shape(tensor), min_value),
      tensor,
  )
  tensor = tf.where(
      tf.math.is_inf(tensor) & tf.math.greater(tensor, 0),
      tf.fill(tf.shape(tensor), max_value),
      tensor,
  )
  # Fix NaN.
  tensor = fix_nan(tensor, min_value)

  # Clip values to the range.
  tensor = tf.clip_by_value(
      tensor, clip_value_min=min_value, clip_value_max=max_value
  )
  return tensor


def project_points(
    points: tf.Tensor,
    extrinsics: tf.Tensor,
    intrinsics: tf.Tensor,
    distortions: tf.Tensor,
) -> tf.Tensor:
  """Perspective projection of points.

  Args:
    points: Points to be projected, (batch_size, num_points, 3).
    extrinsics: Batched camera rotations and translations, (batch_size, 3, 4).
    intrinsics: Batched camera intrinsics parameters, (batch_size, 2, 3).
    distortions: Batched radial and tangential distortions, (batch_size, 5).

  Returns:
    Batched projected points, (batch_size, num_points, 2).
  """

  batch_size, num_points, _ = tf.unstack(tf.shape(points))
  ones = tf.ones([batch_size, num_points, 1], dtype=points.dtype)
  points_homogeneous = tf.concat((points, ones), axis=-1)

  # Transformation from the world to the image coordinate system.
  points_image = extrinsics @ tf.transpose(points_homogeneous, [0, 2, 1])
  points_image = tf.transpose(points_image, [0, 2, 1])

  # Transformation to the undistorted image plane.
  z_coords = points_image[:, :, 2]
  points_image_x = tf.math.divide_no_nan(points_image[:, :, 0], z_coords)
  points_image_y = tf.math.divide_no_nan(points_image[:, :, 1], z_coords)

  k1, k2, k3 = distortions[:, 0], distortions[:, 1], distortions[:, 2]
  p1, p2 = distortions[:, 3], distortions[:, 4]
  r2 = points_image_x**2 + points_image_y**2
  r4 = r2**2
  r6 = r2 * r4
  radial_factor = 1.0 + k1[:, None] * r2 + k2[:, None] * r4 + k3[:, None] * r6

  xy = points_image[:, :, 0] * points_image[:, :, 1]
  # tangential_bias_x = 2*p1*xy + p2
  tangential_bias_x = 2.0 * p1[:, None] * xy + p2[:, None] * (
      r2 + 2.0 * points_image[:, :, 0] ** 2
  )
  # tangential_bias_y = p1*(r2 + 2*points_image[:,1]**2) + 2*p2*xy
  tangential_bias_y = 2.0 * p2[:, None] * xy + p1[:, None] * (
      r2 + 2.0 * points_image[:, :, 1] ** 2
  )

  radial_factor = fix_nan(radial_factor, default_value=1.0)
  tangential_bias_x = fix_nan(tangential_bias_x, default_value=0.0)
  tangential_bias_y = fix_nan(tangential_bias_y, default_value=0.0)

  points_image_x = points_image_x * radial_factor + tangential_bias_x
  points_image_y = points_image_y * radial_factor + tangential_bias_y
  points_image_z = tf.ones_like(points_image_x, dtype=points_image_x.dtype)
  points_image = tf.stack(
      (points_image_x, points_image_y, points_image_z), axis=-1
  )

  # Transformation from distorted image coordinates to the final image
  # coordinates with the camera intrinsics
  points_image = intrinsics @ tf.transpose(points_image, [0, 2, 1])
  points_image = tf.transpose(points_image, [0, 2, 1])
  points_image = fix_and_clip(
      points_image,
      min_value=-_PROJECTION_CLIP_VALUE,
      max_value=_PROJECTION_CLIP_VALUE,
  )
  return points_image
