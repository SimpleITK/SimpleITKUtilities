import SimpleITK as sitk
from PySide6 import QtWidgets
from SimpleITK.utilities.pyside import sitk2qpixmap, qpixmap2sitk


def test_pyside():
    # QPixmap cannot be created without a QGuiApplication
    app = QtWidgets.QApplication()

    # Test grayscale
    scalar_image = sitk.Image([2, 3], sitk.sitkUInt8)
    scalar_image[0, 0] = 1
    scalar_image[0, 1] = 2
    scalar_image[0, 2] = 3
    scalar_image[1, 0] = 253
    scalar_image[1, 1] = 254
    scalar_image[1, 2] = 255

    qpixmap = sitk2qpixmap(scalar_image)
    sitk_image = qpixmap2sitk(qpixmap)
    # Compare on pixel values, metadata information ignored.
    assert sitk.Hash(sitk_image) == sitk.Hash(scalar_image)

    # Test color
    color_image = sitk.Image([2, 3], sitk.sitkVectorUInt8, 3)
    color_image[0, 0] = [0, 1, 2]
    color_image[0, 1] = [4, 8, 16]
    color_image[0, 2] = [32, 64, 128]
    color_image[1, 0] = [0, 10, 20]
    color_image[1, 1] = [30, 40, 50]
    color_image[1, 2] = [60, 70, 80]

    qpixmap = sitk2qpixmap(color_image)
    sitk_image = qpixmap2sitk(qpixmap)
    # Compare on pixel values, metadata information ignored.
    assert sitk.Hash(sitk_image) == sitk.Hash(color_image)
