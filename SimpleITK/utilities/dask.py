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

import dask.array as da
import SimpleITK as sitk
from pathlib import Path
from typing import Union, Tuple

PathType = Union[str, Path]
ChunkType = Union[int, Tuple]


def from_sitk(filename: PathType, chunks: ChunkType = None) -> da.Array:
    """Reads the filename into a dask array with map_block.

    SimpleITK is used to "stream read" chunks from the file if supported, otherwise the entire image will be read for
    each chunk request.ITK support full streaming includes MHA, MRC, NRRD and NIFTI file formats.

    :param filename: A path-like object to the location of an image file readable by SimpleITK.
    :param chunks: Please see dask documentation on chunks of dask arrays for supported formats. Chunk size can be tuned
        for performance based on continuously stored on disk, re-chunking, and downstream processes.
    :return: A Dask array of the image on file.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(filename))
    reader.ReadImageInformation()
    img_shape = reader.GetSize()[::-1]
    # default to loading the whole image from file in case the ITK ImageIO does not support streaming
    if chunks is None:
        chunks = (-1,) * reader.GetDimension()

    is_multi_component = reader.GetNumberOfComponents() != 1

    if is_multi_component:
        img_shape = img_shape + (reader.GetNumberOfComponents(),)

        if len(chunks) < len(img_shape):
            chunks = chunks + (-1,)

    z = da.zeros(
        shape=img_shape, dtype=sitk.extra._get_numpy_dtype(reader), chunks=chunks
    )

    def func(z, block_info=None):
        _reader = sitk.ImageFileReader()
        _reader.SetFileName(str(filename))
        if block_info is not None:
            if is_multi_component:
                size = block_info[None]["chunk-shape"][-2::-1]
                index = [al[0] for al in block_info[None]["array-location"][-2::-1]]
            else:
                size = block_info[None]["chunk-shape"][::-1]
                index = [al[0] for al in block_info[None]["array-location"][::-1]]
            _reader.SetExtractIndex(index)
            _reader.SetExtractSize(size)
            sitk_img = _reader.Execute()
            return sitk.GetArrayFromImage(sitk_img)
        return z

    da_img = da.map_blocks(func, z, meta=z, name="from-sitk")
    return da_img
