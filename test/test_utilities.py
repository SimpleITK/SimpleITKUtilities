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
