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

import itertools
from functools import wraps


def slice_by_slice(func):
    """A function decorator which executes func on each 3D sub-volume and
    *in-place* pastes the results into the input image. The input image type
    and the output image type are required to be the same type.

    :param func: A function which takes a SimpleITK Image as it's first
    argument and returns an Image as the result.

    :return: A decorated function.
    """

    iter_dim = 2

    @wraps(func)
    def _slice_by_slice(image, *args, **kwargs):
        dim = image.GetDimension()

        if dim <= iter_dim:
            #
            image = func(image, *args, **kwargs)
            return image

        extract_size = list(image.GetSize())
        extract_size[iter_dim:] = itertools.repeat(0, dim - iter_dim)

        extract_index = [0] * dim
        paste_idx = [slice(None, None)] * dim

        extractor = sitk.ExtractImageFilter()
        extractor.SetSize(extract_size)

        for high_idx in itertools.product(
            *[range(s) for s in image.GetSize()[iter_dim:]]
        ):
            # The lower 2 elements of extract_index are always 0.
            # The remaining indices are iterated through all indexes.
            extract_index[iter_dim:] = high_idx
            extractor.SetIndex(extract_index)

            # Sliced based indexing for setting image values internally uses
            # the PasteImageFilter executed "in place".  The lower 2 elements
            # are equivalent to ":". For a less general case the assignment
            # could be written as image[:,:,z] = ...
            paste_idx[iter_dim:] = high_idx
            image[paste_idx] = func(extractor.Execute(image), *args, **kwargs)

        return image

    return _slice_by_slice
