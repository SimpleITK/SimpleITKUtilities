import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import vtk.util.numpy_support as vtknp


def sitk2vtk(img, debugOn=False):
    """
    Function to convert a SimpleITK image to a VTK image.
    Written by David T. Chen from the National Institute of Allergy
    and Infectious Diseases, dchen@mail.nih.gov.
    It is covered by the Apache License, Version 2.0:
    http://www.apache.org/licenses/LICENSE-2.0
    """

    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()
    direction = img.GetDirection()

    # there doesn't seem to be a way to specify the image orientation in VTK

    # convert the SimpleITK image to a numpy array
    i2 = sitk.GetArrayFromImage(img)
    if debugOn:
        i2_string = i2.tostring()
        print("data string address inside sitk2vtk", hex(id(i2_string)))

    vtk_image = vtk.vtkImageData()

    # VTK expects 3-dimensional parameters
    if len(size) == 2:
        size.append(1)

    if len(origin) == 2:
        origin.append(0.0)

    if len(spacing) == 2:
        spacing.append(spacing[0])

    if len(direction) == 4:
        direction = [
            direction[0],
            direction[1],
            0.0,
            direction[2],
            direction[3],
            0.0,
            0.0,
            0.0,
            1.0,
        ]

    vtk_image.SetDimensions(size)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    if vtk.vtkVersion.GetVTKMajorVersion() < 9:
        print("Warning: VTK version <9.  No direction matrix.")
    else:
        vtk_image.SetDirectionMatrix(direction)

    # depth_array = numpy_support.numpy_to_vtk(i2.ravel(), deep=True,
    #                                          array_type = vtktype)
    depth_array = numpy_support.numpy_to_vtk(i2.ravel())
    depth_array.SetNumberOfComponents(ncomp)
    vtk_image.GetPointData().SetScalars(depth_array)

    vtk_image.Modified()
    #
    if debugOn:
        print("Volume object inside sitk2vtk")
        print(vtk_image)
        #        print("type = ", vtktype)
        print("num components = ", ncomp)
        print(size)
        print(origin)
        print(spacing)
        print(vtk_image.GetScalarComponentAsFloat(0, 0, 0, 0))

    return vtk_image


def vtk2sitk(vtkimg, debug=False):
    """Takes a VTK image, returns a SimpleITK image."""
    sd = vtkimg.GetPointData().GetScalars()
    npdata = vtknp.vtk_to_numpy(sd)

    dims = list(vtkimg.GetDimensions())
    origin = vtkimg.GetOrigin()
    spacing = vtkimg.GetSpacing()

    if debug:
        print("dims:", dims)
        print("origin:", origin)
        print("spacing:", spacing)

        print("numpy type:", npdata.dtype)
        print("numpy shape:", npdata.shape)

    dims.reverse()
    npdata.shape = tuple(dims)
    if debug:
        print("new shape:", npdata.shape)
    sitkimg = sitk.GetImageFromArray(npdata)
    sitkimg.SetSpacing(spacing)
    sitkimg.SetOrigin(origin)

    direction = vtkimg.GetDirectionMatrix()
    d = []
    for y in range(3):
        for x in range(3):
            d.append(direction.GetElement(y, x))
    sitkimg.SetDirection(d)

    return sitkimg
