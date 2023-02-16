import SimpleITK as sitk


def resize_and_scale(image, new_size, outside_pixel_value=0):
    """
    Resize the given image to the given size, with isotropic pixel spacing
    and scale the intensities to [0,255].

    Resizing retains the original aspect ratio, with the original image centered
    in the new image. Padding is added outside the original image extent using the
    provided value.

    :param image: A SimpleITK image.
    :param new_size: List of ints specifying the new image size.
    :param outside_pixel_value: Value in [0,255] used for padding.
    :param preserve_aspect_ratio:
    :return: a 2D SimpleITK image with desired size and a pixel type of sitkUInt8
    """
    # Rescale intensities if scalar image with pixel type that isn't sitkUInt8.
    # We rescale first, so that the zero padding makes sense for all original image
    # ranges. If we resized first, a value of zero in a high dynamic range image may
    # be somewhere in the middle of the intensity range and the outer border has a
    # constant but arbitrary value.
    if (
        image.GetNumberOfComponentsPerPixel() == 1
        and image.GetPixelID() != sitk.sitkUInt8
    ):
        final_image = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)
    else:
        final_image = image
    new_spacing = [
        ((osz - 1) * ospc) / (nsz - 1)
        for ospc, osz, nsz in zip(
            final_image.GetSpacing(), final_image.GetSize(), new_size
        )
    ]

    new_spacing = [max(new_spacing)] * final_image.GetDimension()

    center = final_image.TransformContinuousIndexToPhysicalPoint(
        [sz / 2.0 for sz in final_image.GetSize()]
    )
    new_origin = [
        c - c_index * nspc
        for c, c_index, nspc in zip(center, [sz / 2.0 for sz in new_size], new_spacing)
    ]
    final_image = sitk.Resample(
        final_image,
        size=new_size,
        outputOrigin=new_origin,
        outputSpacing=new_spacing,
        defaultPixelValue=outside_pixel_value,
    )
    return final_image
