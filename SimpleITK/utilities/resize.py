import SimpleITK as sitk
from typing import Sequence
from typing import Union
from math import ceil
import collections


def _issequence(obj):
    if isinstance(obj, (bytes, str)):
        return False
    return isinstance(obj, collections.abc.Sequence)


def resize(
    image: sitk.Image,
    new_size: Sequence[int],
    isotropic: bool = True,
    fill: bool = True,
    interpolator=sitk.sitkLinear,
    fill_value: float = 0.0,
    use_nearest_extrapolator: bool = False,
    anti_aliasing_sigma: Union[None, float, Sequence[float]] = None,
) -> sitk.Image:
    """
    Resize an image to an arbitrary size while retaining the original image's spatial location.

    Allows for specification of the target image size in pixels, and whether the image pixels spacing should be
    isotropic. The physical extent of the image's data is retained in the new image, with the new image's spacing
    adjusted to achieve the desired size. The image is centered in the new image.

    :param image: A SimpleITK image.
    :param new_size: The new image size in pixels.
    :param isotropic: If False, the original image is resized to fill the new image size by adjusting space. If True,
    the new image's spacing will be isotropic.
    :param fill: If True, the output image will be new_size, and the original image will be centered in the new image
    with constant or nearest values used to fill in the new image. If False and isotropic is True, the output image's
    new size will be calculated to fit the original image's extent such that at least one dimension is equal to
    new_size.
    :param fill_value: Value used for padding.
    :param interpolator: Interpolator used for resampling.
    :param use_nearest_extrapolator: If True, use a nearest neighbor for extrapolation when resampling, overridding the
    constant fill value.
    :param anti_aliasing_sigma: If zero no antialiasing is performed. If a scalar, it is used as the sigma value in
     physical units for all axes. If None or a sequence, the sigma value for each axis is calculated as
     $sigma = (new_spacing - old_spacing) / 2$ in physical units. Gaussian smoothing is performed prior to resampling
     for antialiasing.
    :return: A SimpleITK image with desired size.
    """
    new_spacing = [
        (osz * ospc) / nsz
        for ospc, osz, nsz in zip(image.GetSpacing(), image.GetSize(), new_size)
    ]

    if isotropic:
        new_spacing = [max(new_spacing)] * image.GetDimension()
    if not fill:
        new_size = [
            ceil(osz * ospc / nspc)
            for ospc, osz, nspc in zip(image.GetSpacing(), image.GetSize(), new_spacing)
        ]

    center_cidx = [0.5 * (sz - 1) for sz in image.GetSize()]
    new_center_cidx = [0.5 * (sz - 1) for sz in new_size]

    new_origin_cidx = [0] * image.GetDimension()
    # The continuous index of the new center of the image, in the original image's continuous index space.
    for i in range(image.GetDimension()):
        new_origin_cidx[i] = center_cidx[i] - new_center_cidx[i] * (
            new_spacing[i] / image.GetSpacing()[i]
        )

    new_origin = image.TransformContinuousIndexToPhysicalPoint(new_origin_cidx)

    input_pixel_type = image.GetPixelID()

    if anti_aliasing_sigma is None:
        # (s-1)/2.0 is the standard deviation of the Gaussian kernel in index space, where s downsample factor defined
        # by nspc/ospc.
        anti_aliasing_sigma = [
            max((nspc - ospc) / 2.0, 0.0)
            for ospc, nspc in zip(image.GetSpacing(), new_spacing)
        ]
    elif not _issequence(anti_aliasing_sigma):
        anti_aliasing_sigma = [anti_aliasing_sigma] * image.GetDimension()

    if any([s < 0.0 for s in anti_aliasing_sigma]):
        raise ValueError("anti_aliasing_sigma must be positive, or None.")
    if len(anti_aliasing_sigma) != image.GetDimension():
        raise ValueError(
            "anti_aliasing_sigma must be a scalar or a sequence of length equal to the image dimension."
        )

    if all([s > 0.0 for s in anti_aliasing_sigma]):
        image = sitk.SmoothingRecursiveGaussian(image, anti_aliasing_sigma)
    else:
        for d, s in enumerate(anti_aliasing_sigma):
            if s > 0.0:
                image = sitk.RecursiveGaussian(image, sigma=s, direction=d)

    return sitk.Resample(
        image,
        size=new_size,
        outputOrigin=new_origin,
        outputSpacing=new_spacing,
        outputDirection=image.GetDirection(),
        defaultPixelValue=fill_value,
        interpolator=interpolator,
        useNearestNeighborExtrapolator=use_nearest_extrapolator,
        outputPixelType=input_pixel_type,
    )
