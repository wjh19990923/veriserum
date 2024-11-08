# -*- coding: utf-8 -*-
"""run using pytest

ToDO: rewrite unittest testcases
figure outputs can be replaced by the usual comparison with a saved pickle
"""
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image
from skimage.io import imread

from .image_tools import (
    check_img,
    create_circ_mask,
    load_img_by_meas_id,
    load_img_by_path,
    save_img_to_server,
)


class Test_create_circ_mask_only_d(unittest.TestCase):
    def setUp(self):
        self.res = create_circ_mask(11)

    def test_shape(self):
        assert self.res.shape == (11, 11)

    def test_middle(self):
        assert self.res[5, 5] == 1

    def test_corners(self):
        for i, j in [(0, 0), (10, 10), (0, 10), (10, 0)]:
            assert self.res[i, j] == 0

    def test_circle_edges(self):
        for i, j in [(10, 5), (0, 5), (5, 0), (10, 5)]:
            assert self.res[i, j] == 1


class Test_create_circ_mask_img_size(unittest.TestCase):
    def setUp(self):
        self.res = create_circ_mask(11, (21, 21))

    def test_shape(self):
        assert self.res.shape == (21, 21)


class Test_create_circ_mask_centre(unittest.TestCase):
    """put a circle in the middle"""

    def setUp(self):
        self.res = create_circ_mask(11, centre=(0, 0))

    def test_shape(self):
        assert self.res.shape == (11, 11)

    def test_middle(self):
        assert self.res[5, 5] == 1

    def test_corner_lt(self):
        assert self.res[0, 0] == 0

    def test_corners(self):
        for i, j in [(10, 10), (0, 10), (10, 0)]:
            assert self.res[i, j] == 0

    def test_circle_edges(self):
        for i, j in [(0, 5), (5, 0)]:
            assert self.res[i, j] == 1

    def test_circle_edges_neg(self):
        for i, j in [(10, 5), (5, 10)]:
            assert self.res[i, j] == 1


class Test_create_circ_mask_real_data(unittest.TestCase):
    """these need to be checked manually"""

    def setUp(self):
        p = Path(__file__).parent
        self.img1 = imread(p / "./testfiles/image_tools_testImg1.tif")

    def test_real(self):
        mask = create_circ_mask(1000)
        plt.figure()
        plt.imshow(self.img1)
        plt.title("Check manually: 1000 pixel, centered on middle, good fit")
        plt.imshow(mask, alpha=0.5)
        plt.show()

    def test_real2(self):
        mask = create_circ_mask(500, img_size=(1000, 1000), centre=(500, 0))
        plt.figure()
        plt.imshow(self.img1)
        plt.title("Check manually: should be diameter 500 pixel centered on lower edge")
        plt.imshow(mask, alpha=0.5)
        plt.show()


def test_load_img_by_path(tmp_path):
    """write an image to disk and read it again
    .. note:: `tmp_path` is pytest fixture to create a temporary directory for the test invocation
    """
    # create a mock object to write img to harddisk, then open
    img = 128 * np.ones((1600, 1664), dtype=np.uint16)
    Image.fromarray(img).save(tmp_path / "testImg.tif")
    res = load_img_by_path(tmp_path / "testImg.tif")
    assert np.array_equal(res, img)


def test_load_img_by_path_doesnt_exist():
    with pytest.raises(FileNotFoundError):
        load_img_by_path("testImg.tif")


def test_load_img_by_path_wrong_size(tmp_path):
    img = 128 * np.ones((160, 1664), dtype=np.uint16)
    Image.fromarray(img).save(tmp_path / "testImg.tif")
    with pytest.raises(AssertionError):
        load_img_by_path(tmp_path / "testImg.tif")


@pytest.mark.parametrize("plane", ["fs", "bs"])
def test_load_img_by_meas_id(tmp_path, plane):
    """imgDir: Path, measId: int, plane: str = "bs", isCalibration=False):"""
    # create a mock object to write img to harddisk, then open
    img = 128 * np.ones((1600, 1664), dtype=np.uint16)
    Image.fromarray(img).save(tmp_path / f"{plane}_000015.tif")
    res = load_img_by_meas_id(imgDir=tmp_path, measId=15, plane=plane)
    assert np.array_equal(res, img)


@pytest.mark.parametrize("plane", ["fs", "bs"])
def test_load_cal_img_by_meas_id(tmp_path, plane):
    """load a calibration measurement"""
    # create a mock object to write img to harddisk, then open
    img = 128 * np.ones((1600, 1664), dtype=np.uint16)
    Image.fromarray(img).save(tmp_path / f"cal_{plane}_000015.tif")
    res = load_img_by_meas_id(imgDir=tmp_path, measId=15, plane=plane, isCalibration=True)
    assert np.array_equal(res, img)


@pytest.mark.parametrize(
    "size",
    [
        1664,
        pytest.param(3, marks=pytest.mark.xfail(raises=AssertionError, strict=True)),
        pytest.param(210, marks=pytest.mark.xfail(raises=AssertionError, strict=True)),
    ],
)
def test_check_img(size):
    """should only work if the size is (1600,1664)"""
    img = 128 * np.ones((1600, size), dtype=np.uint16)
    check_img(img)


allowedImgs = [
    128 * np.ones((1600, 1664), dtype=np.uint16),
    30010 * np.ones((1600, 1664), dtype=np.uint16),
]
params = [(img, p, cal) for p in ["bs", "fs"] for img in allowedImgs for cal in (True, False)]


@pytest.mark.parametrize("img, plane, cal", params)
def test_save_img_to_server(tmp_path, img, plane, cal):
    """save img and check whether it was written with the correct name, also check if you can reopen it using `load_img_by_path`"""
    resPath = save_img_to_server(img, plane, measId=13, outDir=tmp_path, isCalibration=cal)
    assert resPath.exists()
    c = "cal_" if cal else ""
    path = tmp_path / f"{c}{plane}_000013.tif"
    assert (path).exists(), "Image was not written as expected"
    res = load_img_by_path(path)
    assert np.array_equal(res, img), "Image that was loaded did not correspond to the image that was written"


badImgs = [
    128 * np.ones((1600, 1664), dtype=np.int16),
    30010 * np.ones((1600, 1664), dtype=np.float32),
]
params = [(bimg, p, cal) for p in ["bs", "fs"] for bimg in badImgs for cal in (True, False)]


@pytest.mark.parametrize("img, plane, cal", params)
def test_save_img_to_server_wrong_dtype(tmp_path, img, plane, cal):
    """make the test fail if the dtype is not np.uint16"""
    with pytest.raises(AssertionError):
        save_img_to_server(img, plane, measId=13, outDir=tmp_path, isCalibration=cal)


if __name__ == "__main__":
    unittest.main()
