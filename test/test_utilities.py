import SimpleITK as sitk
import SimpleITK.utilities as sitkutils


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
        )
    )
    rgb_hash = sitk.Hash(
        sitkutils.overlay_bounding_boxes(
            image=rgb_image,
            bounding_boxes=bounding_boxes,
            colors=[[255, 20, 147], [255, 215, 0]],
            half_line_width=1,
            bounding_box_format="MINXY_MAXXY",
        )
    )
    assert (
        scalar_hash == "d7dde3eee4c334ffe810a636dff872a6ded592fc"
        and rgb_hash == "d6694a394f8fcc32ea337a1f9531dda6f4884af1"
    )
