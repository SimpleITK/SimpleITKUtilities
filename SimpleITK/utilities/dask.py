import dask.array as da
import SimpleITK as sitk
from pathlib import Path


def from_sitk(filename: Path, chunks=None):
    """
    Reads the filename into a dask array with map_block.

    SimpleITK is used to read chunks from the file if supported, otherwise the entire image will be read for each
    chunk request.

    ITK support full streaming includes MHA, MRC, NRRD and NIFTI file formats.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(filename))
    reader.ReadImageInformation()
    img_shape = reader.GetSize()[::-1]
    # default to loading the whole image from file
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
            print(f"size: {sitk_img.GetSize()}")
            return sitk.GetArrayFromImage(sitk_img)
        return z

    da_img = da.map_blocks(func, z, meta=z, name="from-sitk")
    return da_img
