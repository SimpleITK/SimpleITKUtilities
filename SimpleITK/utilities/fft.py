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


def fft_based_translation_initialization(
    fixed: sitk.Image, moving: sitk.Image
) -> sitk.TranslationTransform:
    """Perform fast Fourier transform based normalized correlation to find the translation which maximizes correlation
    between the images.

    If the moving image grid is not congruent with fixed image ( same origin, spacing and direction ), then it will be
    resampled onto the grid defined by the fixed image.

    Efficiency can be improved by reducing the resolution of the image or using a projection filter to reduce the
    dimensionality of the inputs.

    :param fixed: A SimpleITK image object.
    :param moving: Another SimpleITK Image object, which will be resampled onto the grid of the fixed image if it is not
        congruent.
    :return: A TranslationTransform mapping physical points from the fixed to the moving image.
    """

    if (
        moving.GetSpacing() != fixed.GetSpacing()
        or moving.GetDirection() != fixed.GetDirection()
        or moving.GetOrigin() != fixed.GetOrigin()
    ):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        moving = resampler.Execute(moving)

    sigma = fixed.GetSpacing()[0]
    pixel_type = sitk.sitkFloat32

    fft_fixed = sitk.Cast(sitk.SmoothingRecursiveGaussian(fixed, sigma), pixel_type)
    fft_moving = sitk.Cast(sitk.SmoothingRecursiveGaussian(moving, sigma), pixel_type)

    out = sitk.FFTNormalizedCorrelation(fft_fixed, fft_moving)

    out = sitk.SmoothingRecursiveGaussian(out)
    cc = sitk.ConnectedComponent(sitk.RegionalMaxima(out, fullyConnected=True))
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(out, cc)
    labels = sorted(stats.GetLabels(), key=lambda l: stats.GetMean(l))

    peak_bb = stats.GetBoundingBox(labels[-1])
    # Add 0.5 for center of voxel on continuous index
    peak_idx = [
        (min_idx + max_idx) / 2.0 + 0.5
        for min_idx, max_idx in zip(peak_bb[0::2], peak_bb[1::2])
    ]

    peak_pt = out.TransformContinuousIndexToPhysicalPoint(peak_idx)
    peak_value = stats.GetMean(labels[-1])

    center_pt = out.TransformContinuousIndexToPhysicalPoint(
        [p / 2.0 for p in out.GetSize()]
    )

    translation = [c - p for c, p in zip(center_pt, peak_pt)]

    return sitk.TranslationTransform(out.GetDimension(), translation)
