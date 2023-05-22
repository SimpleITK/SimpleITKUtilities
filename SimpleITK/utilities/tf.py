# ========================================================================
#
#  Copyright NumFOCUS
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ========================================================================

from pathlib import Path
import SimpleITK as sitk
import tensorflow as tf
import numpy as np


def _sitk_image_to_tensor(image: sitk.Image, name=None, channels=3) -> tf.Tensor:
    """Convert an SimpleITK Image to a Tensorflow Tensor.

    :param image: The SimpleITK Image to convert
    :param name: The name of the resulting Tensor
    :param channels: The number of channels to use for the resulting Tensor (default: 3)
    :return: A Tensorflow Tensor

    """

    image = sitk.RescaleIntensity(image, outputMinimum=0.0, outputMaximum=255.0)
    image = sitk.Cast(image, sitk.sitkUInt8)

    if image.GetDimension() == 3 and image.GetSize()[2] == 1:
        image = image[:, :, 0]

    if image.GetDimension() == 2:
        if channels != 1:
            # Add a channel dimension
            # TODO: can be replaced with cast in new version of SimpleITK
            image = sitk.Compose([image] * channels)
    else:
        raise ValueError(f"Unexpected image shape: {image.GetSize()}")
    a = sitk.GetArrayFromImage(image)
    if a.ndim == 2:
        a = np.expand_dims(a, axis=-1)
    tf_image = tf.convert_to_tensor(a, name=name)
    return tf_image


def _sitk_read_image(
    filename: tf.string, image_path=None, encoding="utf-8", channels=3
) -> tf.Tensor:
    """Read an image from a filename into a Tensorflow Tensor of 3 dimensions (height, width, channels).

    The image is converted to float32, rescaled to [0, 255], and the number of channels is adjusted to the parameter
    value.

    If the image is 3D and the third dimension is 1, the image is converted to 2D. This commonly occurs with DICOM
    images.

    If the image does not have the provided number of channels, the image is duplicated along the channel dimension.

    :param filename: The filename to read from
    :param image_path: A directory to prefix all filenames with (optional)
    :param encoding: used for decoding Tensorflow string tensors
    :param channels: The number of channels to use for the resulting Tensor (default: 3)
    :return: A Tensorflow Tensor of the image
    """
    str_fn = filename.numpy().decode(encoding)

    if image_path:
        str_fn = str(Path(image_path) / str_fn)

    image = sitk.ReadImage(str_fn, sitk.sitkFloat32)
    return _sitk_image_to_tensor(image, name=str(Path(str_fn).name), channels=channels)


def dataset_from_dicom_filenames(
    dicom_filenames, image_size, image_path=None, encoding="utf-8", channels=3
) -> tf.data.Dataset:
    """
    Create a Tensorflow Dataset from a list of DICOM filenames. Resizes the images to the provided image size and
    adjusts the number of channels.

    :param dicom_filenames: A list-like of DICOM string filenames
    :param image_size: The (height, width) to resize the image to
    :param image_path: A directory to prefix all filenames with
    :param encoding: used for decoding Tensorflow string tensors
    :param channels: The number of channels to use for the resulting Tensor (default: 3)
    :return: A Tensorflow Dataset with a flow to read and resize the provided image filenames.
    """
    tf_filenames = tf.constant(dicom_filenames, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices(tf_filenames)

    def load_dicom(filename):
        [
            image,
        ] = tf.py_function(
            lambda x: _sitk_read_image(x, image_path, encoding, channels),
            [filename],
            [tf.uint8],
        )
        image.set_shape([None, None, channels])
        image = tf.image.resize_with_pad(
            image,
            image_size[0],
            image_size[1],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=True,
        )
        return image

    return dataset.map(load_dicom)
