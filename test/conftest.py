import pytest
import SimpleITK as sitk
from pathlib import Path
from itertools import product, chain

_params = product(
    [sitk.sitkInt8, sitk.sitkUInt8, sitk.sitkInt16, sitk.sitkUInt16, sitk.sitkFloat32],
    ["nrrd", "nii", "mha"],
)

_params = chain(
    _params,
    product(
        [
            sitk.sitkInt8,
            sitk.sitkInt16,
            sitk.sitkInt16,
            sitk.sitkInt32,
            sitk.sitkFloat32,
        ],
        ["mrc"],
    ),
)


def to_ids(p):
    return f"{p[1]}_{sitk.GetPixelIDValueAsString([0])}"


@pytest.fixture(scope="session", params=_params)
def image_fixture(request, tmp_path_factory) -> Path:
    pixel_type, extension = request.param
    fn = f"image_{sitk.GetPixelIDValueAsString(pixel_type).replace(' ', '_')}.nrrd"
    img = sitk.Image([256, 128, 64], pixel_type)

    fn = Path(tmp_path_factory.mktemp("data")) / fn
    sitk.WriteImage(img, fn)
    return fn


_params = product(
    [sitk.sitkInt8, sitk.sitkUInt8, sitk.sitkInt16, sitk.sitkUInt16, sitk.sitkFloat32],
    ["nrrd", "nii", "mha"],
)
_params = chain(
    _params,
    product(
        [
            sitk.sitkInt8,
            sitk.sitkUInt8,
            sitk.sitkInt16,
            sitk.sitkInt16,
            sitk.sitkInt32,
            sitk.sitkUInt32,
        ],
        ["dcm"],
    ),
)


@pytest.fixture(scope="session", params=_params)
def image_fixture_2d(request, tmp_path_factory) -> Path:
    pixel_type, extension = request.param
    fn = f"image_2d_{sitk.GetPixelIDValueAsString(pixel_type).replace(' ', '_')}.nrrd"
    img = sitk.Image([512, 256], pixel_type)

    fn = Path(tmp_path_factory.mktemp("data")) / fn
    sitk.WriteImage(img, fn)
    return fn
