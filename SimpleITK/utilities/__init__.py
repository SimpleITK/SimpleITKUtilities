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

from .Logger import Logger
from .slice_by_slice import slice_by_slice
from .make_isotropic import make_isotropic
from .fft import fft_based_translation_initialization
from .overlay_bounding_boxes import overlay_bounding_boxes
from .resize import resize

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown version"


__all__ = [
    "Logger",
    "slice_by_slice",
    "make_isotropic",
    "fft_based_translation_initialization",
    "overlay_bounding_boxes",
    "resize",
    "__version__",
]

from importlib.util import find_spec

try:
    find_spec("vtk")
    from .vtk import sitk2vtk, vtk2sitk

    __all__.extend(["sitk2vtk", "vtk2sitk"])

    _has_vtk = True
except ImportError:
    _has_vtk = False

del find_spec
