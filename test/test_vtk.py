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
