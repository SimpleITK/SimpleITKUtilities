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
from SimpleITK.extra import _get_numpy_dtype


def histogram_equalization(
    image: sitk.Image,
    number_of_histogram_levels: int = 256,
    number_of_match_points: int = 32,
    threshold_at_mean_intensity: bool = False,
) -> sitk.Image:
    """
    Histogram equalization is a method in image processing of contrast adjustment using the image's histogram.

    The output intensity range will be [0, number_of_histogram_levels - 1].

    This implementation uses the itk::HistogramMatchingImageFilter to perform the histogram matching with a flat
    histogram. The flat histogram is generated from a reference image created with the specified number of levels
    and then the filter performs histogram matching with the specified number of match points.

    :param image:
    :param number_of_histogram_levels:
    :param number_of_match_points:
    :param threshold_at_mean_intensity:
    :return:
    """

    ramp_values = np.arange(number_of_histogram_levels, dtype=_get_numpy_dtype(image))

    # reshape ramp_values to match the number of dimensions in the image
    ramp_values = ramp_values.reshape((1,) * (image.GetDimension() - 1) + (-1,))

    ramp_image = sitk.GetImageFromArray(ramp_values)

    histigram_matching_filter = sitk.HistogramMatchingImageFilter()
    histigram_matching_filter.SetNumberOfHistogramLevels(number_of_histogram_levels)
    histigram_matching_filter.SetNumberOfMatchPoints(number_of_match_points)
    histigram_matching_filter.SetThresholdAtMeanIntensity(threshold_at_mean_intensity)

    output_image = histigram_matching_filter.Execute(image, ramp_image)

    return output_image
