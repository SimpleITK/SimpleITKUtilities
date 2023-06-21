# ========================================================================
#
#  Copyright NumFOCUS
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ========================================================================

import SimpleITK as sitk
import logging
import vtk
import vtk.util.numpy_support as vtknp

logger = logging.getLogger(__name__)


def sitk2vtk(image: sitk.Image) -> vtk.vtkImageData:
    """Convert a 2D or 3D SimpleITK image to a VTK image. VTK versions prior to
    version 9 do not support a direction cosine matrix. If the installed
    version is lower than that, the direction cosine matrix is ignored and
    that information is lost. A warning is issued using the Python
    logging mechanism.

    :param image: Image to convert.
    :return: A VTK image.
    """

    size = list(image.GetSize())
    if len(size) > 3:
        raise ValueError(
            "Conversion only supports 2D and 3D images, got {len(size)}D image"
        )

    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    ncomp = image.GetNumberOfComponentsPerPixel()

    # VTK expects 3-dimensional image parameters
    if len(size) == 2:
        size.append(1)
        origin = origin + (0.0,)
        spacing = spacing + (1.0,)
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

    # Create VTK image and set its metadata
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(size)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    if vtk.vtkVersion.GetVTKMajorVersion() < 9:
        logger.warning(
            "VTK version <9 does not support direction matrix which is ignored"
        )
    else:
        vtk_image.SetDirectionMatrix(direction)

    # Set pixel data
    depth_array = vtknp.numpy_to_vtk(sitk.GetArrayFromImage(image).ravel())
    depth_array.SetNumberOfComponents(ncomp)
    vtk_image.GetPointData().SetScalars(depth_array)

    vtk_image.Modified()
    return vtk_image


def vtk2sitk(image: vtk.vtkImageData) -> sitk.Image:
    """Convert a VTK image to a  SimpleITK image.

    :param image: Image to convert.
    :return: A SimpleITK image.
    """
    sd = image.GetPointData().GetScalars()
    npdata = vtknp.vtk_to_numpy(sd)
    dims = list(image.GetDimensions())
    dims.reverse()
    npdata.shape = tuple(dims)

    sitk_image = sitk.GetImageFromArray(npdata)
    sitk_image.SetSpacing(image.GetSpacing())
    sitk_image.SetOrigin(image.GetOrigin())
    # By default, direction is identity.

    if vtk.vtkVersion.GetVTKMajorVersion() >= 9:
        # Copy the direction matrix into a list
        dir_mat = image.GetDirectionMatrix()
        direction = [0] * 9
        dir_mat.DeepCopy(direction, dir_mat)
        sitk_image.SetDirection(direction)

    return sitk_image
