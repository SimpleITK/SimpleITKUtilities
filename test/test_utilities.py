import math

import SimpleITK as sitk
import SimpleITK.utilities as sitkutils
from numpy.testing import assert_allclose


def test_Logger():
    logger = sitkutils.Logger()


def test_make_isotropic():
    img = sitk.Image([10, 10, 5], sitk.sitkFloat32)
    img.SetSpacing([0.3, 0.3, 0.6])

    sitkutils.make_isotropic(img)


slice_call = 0


def test_slice_by_slice():
    @sitkutils.slice_by_slice
    def f(_img):
        global slice_call

        _img[:] = slice_call
        slice_call = 1 + slice_call
        return _img

    img = sitk.Image([10, 10, 5], sitk.sitkFloat32)
    img = f(img)

    for z in range(img.GetSize()[2]):
        assert img[0, 0, z] == z


def test_sitktovtk():
    img = sitk.Image([10, 10, 5], sitk.sitkFloat32)
    vtk_img = sitkutils.sitk2vtk(img)


def test_fft_initialization():
    fixed_img = sitk.Image([1024, 512], sitk.sitkInt8)

    fixed_img[510:520, 255:265] = 10

    moving_img = sitk.Image([1024, 512], sitk.sitkInt8)
    moving_img[425:435, 300:320] = 8

    tx = sitkutils.fft_based_translation_initialization(fixed_img, moving_img)
    assert tx.GetOffset() == (-85.0, 50.0)


def test_fft_initialization2():
    fixed_img = sitk.Image([1024, 512], sitk.sitkUInt8)

    fixed_img[510:520, 255:265] = 10

    moving_img = sitk.Image([1024, 512], sitk.sitkUInt8)
    moving_img.SetSpacing([1, 0.5])
    moving_img.SetOrigin([0, -0.25])
    moving_img[425:435, 305:315] = 8

    tx = sitkutils.fft_based_translation_initialization(fixed_img, moving_img)
    assert tx.GetOffset() == (-85.0, -105.0)


def test_overlay_bounding_boxes():
    bounding_boxes = [[10, 10, 60, 20], [200, 180, 230, 250]]
    scalar_image = sitk.Image([256, 256], sitk.sitkUInt8)
    rgb_image = sitk.Compose([scalar_image, scalar_image, scalar_image + 255])

    scalar_hash = sitk.Hash(
        sitkutils.overlay_bounding_boxes(
            image=scalar_image,
            bounding_boxes=bounding_boxes,
            bounding_box_format="MINXY_MAXXY",
        )[0]
    )
    rgb_hash = sitk.Hash(
        sitkutils.overlay_bounding_boxes(
            image=rgb_image,
            bounding_boxes=bounding_boxes,
            colors=[255, 20, 147, 255, 215, 0],
            half_line_width=1,
            bounding_box_format="MINXY_MAXXY",
        )[0]
    )
    assert (
        scalar_hash == "d7dde3eee4c334ffe810a636dff872a6ded592fc"
        and rgb_hash == "d6694a394f8fcc32ea337a1f9531dda6f4884af1"
    )


def test_resize():
    original_image = sitk.Image([128, 128], sitk.sitkUInt8) + 50
    resized_image = sitkutils.resize(image=original_image, new_size=[128, 128])
    assert resized_image.GetSize() == (128, 128)
    assert resized_image.GetSpacing() == (1.0, 1.0)
    assert resized_image.GetOrigin() == (0.0, 0.0)
    assert resized_image.TransformContinuousIndexToPhysicalPoint((-0.5, -0.5)) == (
        -0.5,
        -0.5,
    )

    resized_image = sitkutils.resize(image=original_image, new_size=[64, 64])
    assert resized_image.GetSize() == (64, 64)
    assert resized_image.GetSpacing() == (2.0, 2.0)
    assert resized_image.GetOrigin() == (0.5, 0.5)
    assert resized_image.TransformContinuousIndexToPhysicalPoint((-0.5, -0.5)) == (
        -0.5,
        -0.5,
    )

    resized_image = sitkutils.resize(
        image=original_image, new_size=[64, 128], fill=False
    )
    assert resized_image.GetSize() == (64, 64)
    assert resized_image.GetSpacing() == (2.0, 2.0)
    assert resized_image.GetOrigin() == (0.5, 0.5)
    assert resized_image.TransformContinuousIndexToPhysicalPoint((-0.5, -0.5)) == (
        -0.5,
        -0.5,
    )

    resized_image = sitkutils.resize(
        image=original_image, new_size=[64, 128], isotropic=False
    )
    assert resized_image.GetSize() == (64, 128)
    assert resized_image.GetSpacing() == (2.0, 1.0)
    assert resized_image.GetOrigin() == (0.5, 0.0)
    assert resized_image.TransformContinuousIndexToPhysicalPoint((-0.5, -0.5)) == (
        -0.5,
        -0.5,
    )


def test_resize_fill():
    original_image = sitk.Image([16, 32], sitk.sitkFloat32) + 1.0

    resized_image = sitkutils.resize(
        image=original_image, new_size=[32, 32], fill=True, fill_value=10.0
    )
    assert resized_image.GetSize() == (32, 32)
    assert resized_image.GetSpacing() == (1.0, 1.0)
    assert resized_image.GetOrigin() == (-8.0, 0.0)
    assert resized_image[0, 0] == 10.0
    assert resized_image[15, 15] == 1.0
    assert resized_image[31, 31] == 10.0

    resized_image = sitkutils.resize(
        image=original_image,
        new_size=[32, 32],
        fill=True,
        use_nearest_extrapolator=True,
    )
    assert resized_image.GetSize() == (32, 32)
    assert resized_image.GetSpacing() == (1.0, 1.0)
    assert resized_image.GetOrigin() == (-8.0, 0.0)
    assert resized_image[0, 0] == 1.0
    assert resized_image[15, 15] == 1.0
    assert resized_image[31, 31] == 1.0


def test_resize_anti_aliasing():
    original_image = sitk.Image([5, 5], sitk.sitkFloat32)
    original_image[2, 2] = 1.0

    resized_image = sitkutils.resize(
        image=original_image,
        new_size=[3, 3],
        interpolator=sitk.sitkNearestNeighbor,
        anti_aliasing_sigma=0,
    )
    assert resized_image.GetSize() == (3, 3)
    assert_allclose(resized_image.GetSpacing(), (5 / 3, 5 / 3))
    assert_allclose(resized_image.GetOrigin(), (1 / 3, 1 / 3))
    assert resized_image[0, 0] == 0.0
    assert resized_image[1, 1] == 1.0
    assert resized_image[1, 0] == 0.0
    assert resized_image[0, 1] == 0.0

    resized_image = sitkutils.resize(
        image=original_image,
        new_size=[3, 3],
        interpolator=sitk.sitkNearestNeighbor,
        anti_aliasing_sigma=None,
    )
    assert resized_image.GetSize() == (3, 3)
    assert_allclose(resized_image.GetSpacing(), (5 / 3, 5 / 3))
    assert_allclose(resized_image.GetOrigin(), (1 / 3, 1 / 3))
    assert math.isclose(resized_image[1, 1], 0.960833, abs_tol=1e-6)
    assert resized_image[1, 0] == resized_image[0, 1]
    assert resized_image[0, 0] == resized_image[2, 2]

    resized_image = sitkutils.resize(
        image=original_image,
        new_size=[3, 3],
        interpolator=sitk.sitkNearestNeighbor,
        anti_aliasing_sigma=0.5,
    )
    assert resized_image.GetSize() == (3, 3)
    assert_allclose(resized_image.GetSpacing(), (5 / 3, 5 / 3))
    assert_allclose(resized_image.GetOrigin(), (1 / 3, 1 / 3))
    assert math.isclose(resized_image[1, 1], 0.621714, abs_tol=1e-6)
    assert math.isclose(resized_image[0, 0], 0, abs_tol=1e-6)
    assert math.isclose(resized_image[1, 0], resized_image[0, 1], abs_tol=1e-8)
    assert math.isclose(resized_image[0, 0], resized_image[2, 2], abs_tol=1e-8)

    resized_image = sitkutils.resize(
        image=original_image,
        new_size=[3, 3],
        interpolator=sitk.sitkNearestNeighbor,
        anti_aliasing_sigma=[1.0, 0.0],
    )
    assert resized_image.GetSize() == (3, 3)
    assert_allclose(resized_image.GetSpacing(), (5 / 3, 5 / 3))
    assert_allclose(resized_image.GetOrigin(), (1 / 3, 1 / 3))
    assert math.isclose(resized_image[1, 1], 0.400101, abs_tol=1e-6)
    assert math.isclose(resized_image[0, 0], 0, abs_tol=1e-6)
    assert resized_image[0, 0] == resized_image[2, 2]
    assert resized_image[1, 0] == 0.0
    assert resized_image[1, 2] == 0.0
