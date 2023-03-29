import SimpleITK as sitk
from typing import Sequence
from typing import Union
from math import ceil


def resize(
    image: sitk.Image,
    new_size: Sequence[int],
    isotropic: bool = True,
    fill: bool = True,
    interpolator=sitk.sitkLinear,
    outside_pixel_value: Union[int, float] = 0,
    use_nearest_extrapolator=False,
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
    :param outside_pixel_value: Value used for padding.
    :param interpolator: Interpolator used for resampling.
    :param use_nearest_extrapolator: If True, use a nearest neighbor extrapolator when resampling, instead of
    constant fill value.
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

    return sitk.Resample(
        image,
        size=new_size,
        outputOrigin=new_origin,
        outputSpacing=new_spacing,
        outputDirection=image.GetDirection(),
        defaultPixelValue=outside_pixel_value,
        interpolator=interpolator,
        useNearestNeighborExtrapolator=use_nearest_extrapolator,
    )
