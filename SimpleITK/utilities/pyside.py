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

import SimpleITK as sitk
import numpy as np
from PySide6.QtGui import QPixmap, QImage


def sitk2qimage(image: sitk.Image) -> QImage:
    """Convert a SimpleITK.Image to PySide QImage.

    Works with 2D images, grayscale or three channel, where it is assumed that
    the three channels represent values in the RGB color space. In SimpleITK there is no notion
    of color space, so if the three channels are in the HSV colorspace the display
    will look strange. If the SimpleITK pixel type represents a high dynamic range,
    the intensities are linearly scaled to [0,255].

    :param image: Image to convert.
    :return: A QImage.
    """
    number_components_per_pixel = image.GetNumberOfComponentsPerPixel()
    if number_components_per_pixel not in [1, 3]:
        raise ValueError(
            f"SimpleITK image has {number_components_per_pixel} channels, expected 1 or 3 channels"
        )
    if number_components_per_pixel == 3 and image.GetPixelID() != sitk.sitkVectorUInt8:
        raise ValueError(
            f"SimpleITK three channel image has pixel type ({image.GetPixelIDTypeAsString()}), expected vector 8-bit unsigned integer"
        )

    if number_components_per_pixel == 1 and image.GetPixelID() != sitk.sitkUInt8:
        image = sitk.Cast(
            sitk.RescaleIntensity(image, outputMinimum=0, outputMaximum=255),
            sitk.sitkUInt8,
        )
    arr = sitk.GetArrayViewFromImage(image)
    return QImage(
        arr.data,
        image.GetWidth(),
        image.GetHeight(),
        arr.strides[0],  # number of bytes per row
        QImage.Format_Grayscale8
        if number_components_per_pixel == 1
        else QImage.Format_RGB888,
    )


def sitk2qpixmap(image: sitk.Image) -> QPixmap:
    """Convert a SimpleITK.Image to PySide QPixmap.

    Works with 2D images, grayscale or three channel, where it is assumed that
    the three channels represent values in the RGB color space. In SimpleITK there is no notion
    of color space, so if the three channels are in the HSV colorspace the display
    will look strange. If the SimpleITK pixel type represents a high dynamic range,
    the intensities are linearly scaled to [0,255].

    :param image: Image to convert.
    :return: A QPixmap.
    """
    return QPixmap.fromImage(sitk2qimage(image))


def qimage2sitk(image: QImage) -> sitk.Image:
    """Convert a QImage to SimpleITK.Image.

    If the QImage contains a grayscale image will return a scalar
    SimpleITK image. Otherwise, returns a three channel RGB image.

    :param image: QImage to convert.
    :return: A SimpleITK image, single channel or three channel RGB.
    """
    # Use constBits() to get the raw data without copying (bits() returns a deep copy).
    # Then reshape the array to the image shape.
    is_vector = True
    # Convert image to Format_RGB888 because it keeps the byte order
    # regardless of big/little endian (RGBA8888 doesn't).
    image = image.convertToFormat(QImage.Format_RGB888)
    arr = np.ndarray(
        (image.height(), image.width(), 3),
        buffer=image.constBits(),
        strides=[image.bytesPerLine(), 3, 1],
        dtype=np.uint8,
    )
    if image.isGrayscale():
        arr = arr[:, :, 0]
        is_vector = False
    return sitk.GetImageFromArray(arr, isVector=is_vector)


def qpixmap2sitk(pixmap: QPixmap) -> sitk.Image:
    """Convert a QPixmap to SimpleITK.Image.

    If the QPixmap contains a grayscale image will return a scalar
    SimpleITK image. Otherwise, returns a four channel RGBA image.

    :param qpixmap: QPixmap to convert.
    :return: A SimpleITK image, single channel or four channel.
    """
    return qimage2sitk(pixmap.toImage())
