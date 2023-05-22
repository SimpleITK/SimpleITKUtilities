import pytest
import SimpleITK as sitk
from SimpleITK.utilities.tf import dataset_from_dicom_filenames
from pathlib import Path


@pytest.fixture
def dataset_directory(tmp_path):
    dataset = [Path(tmp_path, f"test_{i}.dcm") for i in range(10)]

    for idx, filenames in enumerate(dataset):
        # Create a dummy DICOM file from a simpleitk image
        img = sitk.Image(128, 128, sitk.sitkInt16)
        img.SetSpacing([1.0, 1.0])
        img.SetOrigin([0.0, 0.0])
        img.SetDirection([1.0, 0.0, 0.0, 1.0])

        # Set pixel values in img
        for x in range(128):
            for y in range(128):
                img[x, y] = x + y + idx

        sitk.WriteImage(img, str(filenames))

    return dataset


def test_dataset_from_dicom_filenames(dataset_directory):
    dataset = dataset_from_dicom_filenames(
        [str(fn) for fn in dataset_directory], image_size=(128, 128, 1)
    )

    for element in dataset:
        print(element)
