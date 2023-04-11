import pytest
import SimpleITK as sitk
from SimpleITK.utilities.dask import from_sitk
import numpy as np


def test_from_sitk(image_fixture):
    sitk_img = sitk.ReadImage(image_fixture)

    img = from_sitk(image_fixture)
    img.compute()
    assert img.shape == sitk_img.GetSize()[::-1]
    assert np.array_equal(np.asarray(img), sitk.GetArrayViewFromImage(sitk_img))
    assert img.dtype == sitk.extra._get_numpy_dtype(sitk_img)

    img = from_sitk(image_fixture, chunks=(1, -1, -1))
    print(f"chunks: {[c[0] for c in img.chunks]}")
    img.compute()
    assert [c[0] for c in img.chunks] == [
        1,
        sitk_img.GetSize()[1],
        sitk_img.GetSize()[0],
    ]
    assert np.array_equal(np.asarray(img), sitk.GetArrayViewFromImage(sitk_img))
    assert img.dtype == sitk.extra._get_numpy_dtype(sitk_img)

    img = from_sitk(image_fixture, chunks=(1, 128, 128))
    img.compute()
    assert [c[0] for c in img.chunks] == [1, 128, 128]
    assert np.array_equal(np.asarray(img), sitk.GetArrayViewFromImage(sitk_img))
    assert img.dtype == sitk.extra._get_numpy_dtype(sitk_img)

    img = from_sitk(image_fixture, chunks=(-1, 83, 89))
    img.compute()
    assert [c[0] for c in img.chunks] == [sitk_img.GetSize()[2], 83, 89]
    assert np.array_equal(np.asarray(img), sitk.GetArrayViewFromImage(sitk_img))
    assert img.dtype == sitk.extra._get_numpy_dtype(sitk_img)


def test_from_sitk_2d(image_fixture_2d):
    sitk_img = sitk.ReadImage(image_fixture_2d)

    img = from_sitk(image_fixture_2d)
    img.compute()
    assert img.shape == sitk_img.GetSize()[::-1]
    assert np.array_equal(np.asarray(img), sitk.GetArrayViewFromImage(sitk_img))
    assert img.dtype == sitk.extra._get_numpy_dtype(sitk_img)

    img = from_sitk(image_fixture_2d, chunks=(128, -1))
    img.compute()
    assert [c[0] for c in img.chunks] == [128, sitk_img.GetSize()[0]]
    assert np.array_equal(np.asarray(img), sitk.GetArrayViewFromImage(sitk_img))
    assert img.dtype == sitk.extra._get_numpy_dtype(sitk_img)

    img = from_sitk(image_fixture_2d, chunks=(128, 128))
    img.compute()
    assert [c[0] for c in img.chunks] == [128, 128]
    assert np.array_equal(np.asarray(img), sitk.GetArrayViewFromImage(sitk_img))
    assert img.dtype == sitk.extra._get_numpy_dtype(sitk_img)
