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
import itertools


def make_isotropic(
    image,
    interpolator=sitk.sitkLinear,
    spacing=None,
    default_value=0,
    standardize_axes=False,
):
    """
    Many file formats (e.g. jpg, png,...) expect the pixels to be isotropic, same
    spacing for all axes. Saving non-isotropic data in these formats will result in
    distorted images. This function makes an image isotropic via resampling, if needed.
    Args:
        image (SimpleITK.Image): Input image.
        interpolator: By default the function uses a linear interpolator. For
                      label images one should use the sitkNearestNeighbor interpolator
                      so as not to introduce non-existent labels.
        spacing (float): Desired spacing. If none given then use the smallest spacing from
                         the original image.
        default_value (image.GetPixelID): Desired pixel value for resampled points that fall
                                          outside the original image (e.g. HU value for air, -1000,
                                          when image is CT).
        standardize_axes (bool): If the original image axes were not the standard ones, i.e. non
                                 identity cosine matrix, we may want to resample it to have standard
                                 axes. To do that, set this parameter to True.
    Returns:
        SimpleITK.Image with isotropic spacing which occupies the same region in space as
        the input image.
    """
    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    if spacing is None:
        spacing = min(original_spacing)
    new_spacing = [spacing] * image.GetDimension()
    new_size = [
        int(round(osz * ospc / spacing))
        for osz, ospc in zip(original_size, original_spacing)
    ]
    new_direction = image.GetDirection()
    new_origin = image.GetOrigin()
    # Only need to standardize axes if user requested and the original
    # axes were not standard.
    if standardize_axes and not np.array_equal(
        np.array(new_direction), np.identity(image.GetDimension()).ravel()
    ):
        new_direction = np.identity(image.GetDimension()).ravel()
        # Compute bounding box for the original, non standard axes image.
        boundary_points = []
        for boundary_index in list(
            itertools.product(*zip([0] * image.GetDimension(), image.GetSize()))
        ):
            boundary_points.append(image.TransformIndexToPhysicalPoint(boundary_index))
        max_coords = np.max(boundary_points, axis=0)
        min_coords = np.min(boundary_points, axis=0)
        new_origin = min_coords
        new_size = (((max_coords - min_coords) / spacing).round().astype(int)).tolist()
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        new_origin,
        new_spacing,
        new_direction,
        default_value,
        image.GetPixelID(),
    )
