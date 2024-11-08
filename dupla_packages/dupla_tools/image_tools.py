"""Xray images are saved as 16-bit tiffs on different locations on the server
name of the xray images are fs_{6 digit measurement id} (or bs_)
"""
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def check_img(im: np.array):
    """quality checks on the loaded image"""
    assert im.shape == (
        1600,
        1664,
    ), f"Unexpected image size {im.shape} instead of 1600, 1664"
    assert im.dtype == np.uint16, f"Unexpected image datatype {im.dtype} instead of np.uint16"


def image_name_from_idx(idx: int, plane: str, cal=False) -> str:
    """formats the name of the image, or the calibration image"""
    assert idx < 10e6, "Only support 6 digits"
    assert plane in ["bs", "fs"], "Unsupported plane argument passed"
    cal = "cal_" if cal else ""
    return f"{cal}{plane}_{str(idx):0>6}.tif"


def load_img_by_meas_id(imgDir: Path, measId: int, plane: str = "bs", isCalibration=False):
    """derive image name from `measId`, load as numpy array and performs quality checks
    Args:
        imgDir - the directory to look in
        isCalibration - if True, then prepend "cal_"
    Raises:
         FileNotFoundError if file is not found (duh)
         AssertionError if a quality check fails
    """
    p = imgDir / image_name_from_idx(measId, plane, cal=isCalibration)
    return load_img_by_path(p)


def load_img_by_path(p: Path) -> np.array:
    """load img by a given path, perform quality checks, return np.array
    .. warning:: datatype is np.uint16"""
    im = Image.open(p)
    im = np.array(im, dtype=np.uint16)
    check_img(im)
    return im


def save_img_to_server(
    imgData: np.ndarray,
    intensifier: str,
    measId: int,
    outDir: Path,
    isCalibration=False,
    plot=False,
    check_only=False,
) -> Path:
    """save the np.array to a .tif file under directory `outDir`
    Args:
        imgData: the actual data as array, expected to ne np.uint16
        intensifier: either "bs" or "fs"
        measId: the `measurements.id` from the database
        outDir: where to save the image
        isCalibration: is the image a calibration image, then we prepend cal_
        plot: show plot for debug reasons
        check_only: only check if these transfers would be possible
    Returns:
        complete Path to written file, with suffix ".tif" and the ID in 6 digits. E.g. fs, 2 becomes fs_000002.tif
    Raises:
        OSError if the file could not be written because it already existed
    Example:
        filename would be `fs_000010.tif` or `cal_bs_101023.tif`
    .. warning:: we are truncating to 6 numbers
    """
    outDir = Path(outDir)
    assert imgData.dtype == np.uint16
    if intensifier == "fs" or intensifier == "bs":
        name = image_name_from_idx(measId, intensifier, isCalibration)
    else:
        raise ValueError("Intensifier must be fs or bs")
    outPath = outDir / name
    # creates directories if needed
    Path(outPath.parent).mkdir(parents=True, exist_ok=True)
    if outPath.exists():
        raise OSError(17, f"Image {outPath} already exists. This should not happen.")
    if check_only:
        # additional checks
        assert outDir.is_file() is False, f"The given outDir {outDir} points to a file instead of a directory"
        assert os.access(outDir, os.W_OK), f"Parent directory {outDir} is not writable"
    else:
        Image.fromarray(imgData).save(outPath)
    if plot:
        plt.set_cmap("gray")
        plt.imshow(imgData)
        plt.suptitle(image_name_from_idx(measId, intensifier, isCalibration))
        plt.show()
    return outPath


def create_circ_mask(d: int, img_size: tuple[int, int] = None, centre: tuple[int, int] = None) -> np.array:
    """
    Creates a circular mask of 1s inside, rest set to 0 outside
    Typically used to mask pixels outside the image intensifier for fluoroscope
    images

    Args:
        d: int or None, diameter of the circle;
        img_size: int, int, width and height of the total mask
            None - set to d
        centre: tuple (int,int) or None;
            0/0 is in the middle of the image,
            first number is vertical (positive going down)
            second number is horizontal (positive to the right)
            None - centre is put in the middle of the mask
    Returns:
        disk mask: numpy array (2D), filled with 0s and 1s
    """
    # set defaults
    size = (d, d) if img_size is None else img_size
    c = (0, 0) if centre is None else centre
    # create circular mask centered at x=c[0] and y=c[1] and radius r
    row, col = np.ogrid[: size[0], : size[1]]
    disk_mask = (row - c[0] - size[0] // 2) ** 2 + (col - c[1] - size[1] // 2) ** 2 < (d / 2) ** 2
    return disk_mask

