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
    fixed: sitk.Image,
    moving: sitk.Image,
    *,
    required_fraction_of_overlapping_pixels: float = 0.0,
    initial_transform: sitk.Transform = None,
    masked_pixel_value: float = None,
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
    :param required_fraction_of_overlapping_pixels: The required fraction of overlapping pixels between the fixed and
        moving image. The value should be in the range of [0, 1]. If the value is 1, then the full overlap is required.
    :param initial_transform: An initial transformation to be applied to the moving image by resampling before the
        FFT registration. The returned transform will be of the initial_transform type with the translation updated.
    :param masked_pixel_value: The value of input pixels to be ignored by correlation. If None, then the
        FFTNormalizedCoorrelation will be used, otherwise the MaskedFFTNormalizedCorrelation will be used.
    :return: A TranslationTransform (or the initial_transform tyype) mapping physical points from the fixed to the
     moving image.
    """

    if (
        initial_transform is not None
        or moving.GetSpacing() != fixed.GetSpacing()
        or moving.GetDirection() != fixed.GetDirection()
        or moving.GetOrigin() != fixed.GetOrigin()
    ):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)

        if initial_transform is not None:
            resampler.SetTransform(initial_transform)
        moving = resampler.Execute(moving)

    sigma = fixed.GetSpacing()[0]
    pixel_type = sitk.sitkFloat32

    fixed = sitk.Cast(sitk.SmoothingRecursiveGaussian(fixed, sigma), pixel_type)
    moving = sitk.Cast(sitk.SmoothingRecursiveGaussian(moving, sigma), pixel_type)

    if masked_pixel_value is None:
        xcorr = sitk.FFTNormalizedCorrelation(
            fixed,
            moving,
            requiredFractionOfOverlappingPixels=required_fraction_of_overlapping_pixels,
        )
    else:
        xcorr = sitk.MaskedFFTNormalizedCorrelation(
            fixed,
            moving,
            sitk.Cast(fixed != masked_pixel_value, pixel_type),
            sitk.Cast(moving != masked_pixel_value, pixel_type),
            requiredFractionOfOverlappingPixels=required_fraction_of_overlapping_pixels,
        )

    xcorr = sitk.SmoothingRecursiveGaussian(xcorr, sigma)

    cc = sitk.ConnectedComponent(sitk.RegionalMaxima(xcorr, fullyConnected=True))
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(xcorr, cc)
    labels = sorted(stats.GetLabels(), key=lambda l: stats.GetMean(l))

    peak_bb = stats.GetBoundingBox(labels[-1])
    # Add 0.5 for center of voxel on continuous index
    peak_idx = [
        (min_idx + max_idx) / 2.0 + 0.5
        for min_idx, max_idx in zip(peak_bb[0::2], peak_bb[1::2])
    ]

    peak_pt = xcorr.TransformContinuousIndexToPhysicalPoint(peak_idx)
    peak_value = stats.GetMean(labels[-1])

    center_pt = xcorr.TransformContinuousIndexToPhysicalPoint(
        [p / 2.0 for p in xcorr.GetSize()]
    )

    translation = [c - p for c, p in zip(center_pt, peak_pt)]
    if initial_transform is not None:
        offset = initial_transform.TransformVector(translation, point=[0, 0])

        tx_out = sitk.Transform(initial_transform).Downcast()
        tx_out.SetTranslation(
            [a + b for (a, b) in zip(initial_transform.GetTranslation(), offset)]
        )
        return tx_out

    return sitk.TranslationTransform(xcorr.GetDimension(), translation)
