import SimpleITK as sitk
from SimpleITK.utilities.vtk import sitk2vtk, vtk2sitk
import vtk
import gc


def test_sitktovtk():
    img = sitk.Image([10, 10, 5], sitk.sitkFloat32)
    img = img + 42.0
    vtk_img = sitk2vtk(img)

    # free the SimpleITK image's memory
    img = None
    gc.collect()

    assert vtk_img.GetScalarComponentAsFloat(0, 0, 0, 0) == 42.0


def test_vtktositk():
    source = vtk.vtkImageSinusoidSource()
    source.Update()
    img = source.GetOutput()

    sitkimg = vtk2sitk(img)
    source = None
    img = None
    gc.collect()

    assert sitkimg[0, 0, 0] == 255.0

def test_multichannel():
    img = sitk.Image([10, 10], sitk.sitkVectorUInt8, 3)
    img[0,0] = (255, 127, 42)
    vtk_img = sitk2vtk(img)
    print(vtk_img)

    assert vtk_img.GetNumberOfScalarComponents() == 3

    r = vtk_img.GetScalarComponentAsFloat(0, 0, 0, 0)
    g = vtk_img.GetScalarComponentAsFloat(0, 0, 0, 1)
    b = vtk_img.GetScalarComponentAsFloat(0, 0, 0, 2)

    assert (r, g, b) == (255, 127, 42)
