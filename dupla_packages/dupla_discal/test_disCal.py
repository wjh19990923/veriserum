"""
# as tests, create synthetic grid with different subpixel translations
# then add other distortions
"""
import numpy as np
import pandas as pd
from pytest import approx
from pandas.testing import assert_frame_equal
from PIL import Image
import matplotlib.pyplot as plt
from numpy.testing import assert_array_almost_equal
from pathlib import Path

from .disCal import (
    GridCalibration_GridROI_COMExtractor,
    GridCalibration_HoughROI_COMExtractor,
    GridCalibration_HoughROI_PassThrough_PointExtractor,
    GridCalibrationRadial_HoughROI_PassThrough_PointExtractor,
    GridCalibrationDivision_HoughROI_PassThrough_PointExtractor,
    GridCalibrationDivisionLogistic_HoughROI_PassThrough_PointExtractor,
)
from .extract_points import CenterOfMass_PointExtractor as COM
from .rois import RegionOfInterest


def test_calculate_center_of_mass_square():
    img = np.ones((4, 4))
    xCOM, yCOM = COM.calculate_center_of_mass(img)
    xTrue, yTrue = (1.5, 1.5)
    assert xCOM == xTrue
    assert yCOM == yTrue


def test_calculate_center_of_mass_square_grey():
    img = np.ones((4, 4))
    img[:, 0] = 0.0
    xCOM, yCOM = COM.calculate_center_of_mass(img)
    xTrue, yTrue = (2.0, 1.5)
    assert xCOM == xTrue
    assert yCOM == yTrue


def test_calculate_center_of_mass_rectangular():
    """non-square image"""
    img = np.ones((10, 15))
    xCOM, yCOM = COM.calculate_center_of_mass(img)
    # half of 15 and 10 elements, but starting at 0
    xTrue, yTrue = (15 / 2 - 0.5, 10 / 2 - 0.5)
    assert xCOM == xTrue
    assert yCOM == yTrue


def test_calculate_center_of_mass_rectanguar_grey():
    """rectanglar with a gradient"""
    img = np.ones((10, 15))
    img[0, :] = 0.0
    img[:, 0] = 0.0
    xCOM, yCOM = COM.calculate_center_of_mass(img)
    xTrue, yTrue = (7.5, 5.0)
    assert xCOM == xTrue
    assert yCOM == yTrue


def test_calculate_center_of_mass_complex():
    img = np.zeros((16, 16))
    img[3, 3] = 1.0
    img[8, 0] = 1.0
    img[2, 5] = 0.4
    xCOM, yCOM = COM.calculate_center_of_mass(img)
    xTrue, yTrue = (2.083333, 4.916666)
    assert approx(xCOM) == xTrue
    assert approx(yCOM) == yTrue


def test_forward():
    """do not transform, image should stay the same"""
    gc = GridCalibration_GridROI_COMExtractor(polynomialN=1)
    gc._R = np.eye(2)
    gc._T = np.array([0, 0])
    gc._S = 1.0 
    gc.reflectY = False
    # only linear in x on the x-and coord linear in y on the y coord
    gc._params = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    df = pd.DataFrame([(1, 2), (4, 4)], columns=["x", "y"])
    new = gc.forward(df)
    assert_array_almost_equal(new, df)


def test_forward2():
    """test a specific linear transform in both components
    x direction:  1*x + 2*xy
    y direction: 1*y + 1*x
    """
    gc = GridCalibration_GridROI_COMExtractor(polynomialN=1)
    gc._R = np.eye(2)
    gc._T = np.array([0, 0])
    gc._S = 1.0 
    gc.reflectY = False
    gc._params = np.array([0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0, 0.0])
    df = pd.DataFrame([(1, 2), (4, 5)], columns=["x", "y"])
    new = gc.forward(df)
    expected = pd.DataFrame([(1 * 1 + 2 * 1 * 2, 1 + 2), (1 * 4 + 2 * 4 * 5, 5 + 4)])
    assert_array_almost_equal(new, expected)


def test_forward_shift():
    """shift by 1,2 in x,y"""
    gc = GridCalibration_GridROI_COMExtractor(polynomialN=1)
    gc._R = np.eye(2)
    gc._T = np.array([0, 0])
    gc._S = 1.0 
    gc.reflectY = False
    # only linear in x on the x-and coord linear in y on the y coord
    gc._params = np.array([1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0])
    df = pd.DataFrame([(1, 2), (4, 4)], columns=["x", "y"])
    new = gc.forward(df)
    expected = pd.DataFrame([(2, 4), (5, 6)], columns=["x", "y"])
    assert_array_almost_equal(new.values, expected.values)


def test_forward_shift2():
    """test a specific linear transform in both components
    x direction:  5+ 1*x + 2*xy
    y direction: 7+ 1*y + 1*x
    """
    gc = GridCalibration_GridROI_COMExtractor(polynomialN=1)
    gc._R = np.eye(2)
    gc._T = np.array([0, 0])
    gc._S = 1.0 
    gc.reflectY = False
    gc._params = np.array([5.0, 0.0, 1.0, 2.0, 7.0, 1.0, 1.0, 0.0])
    df = pd.DataFrame([(1, 2), (4, 5)], columns=["x", "y"])
    new = gc.forward(df)
    expected = pd.DataFrame(
        [(5 + 1 * 1 + 2 * 1 * 2, 7 + 1 + 2), (5 + 1 * 4 + 2 * 4 * 5, 7 + 5 + 4)]
    )
    assert_array_almost_equal(new, expected)


def test_forward_shift_higher_polynomial():
    """shift by 1,2 in x,y"""
    gc = GridCalibration_GridROI_COMExtractor(polynomialN=2)
    gc._R = np.eye(2)
    gc._T = np.array([0, 0])
    gc._S = 1.0 
    gc.reflectY = False
    # only linear in x on the x-and coord linear in y on the y coord
    # ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2))
    gc._params = np.array(
        [
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    df = pd.DataFrame([(1, 2), (4, 4)], columns=["x", "y"])
    new = gc.forward(df)
    expected = pd.DataFrame([(2, 4), (5, 6)], columns=["x", "y"])
    assert_array_almost_equal(new.values, expected.values)


def test_loss():
    """put in perfect predictions, loss should be log(0) = inf"""
    gc = GridCalibration_GridROI_COMExtractor(polynomialN=1)
    gc._R = np.eye(2)
    gc._T = np.array([0, 0])
    gc._S = 1.0 
    gc.reflectY = False
    freeParams = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    measured = gc.truePoints.copy()
    res = gc.loss(freeParams, gc.truePoints, measured)
    assert np.isneginf(res)


def test_loss2():
    """put in perfect predictions shifted by 1, loss should be 1 times number of points"""
    gc = GridCalibration_GridROI_COMExtractor(polynomialN=1)
    gc._R = np.eye(2)
    gc._T = np.array([0, 0])
    gc._S = 1.0 
    gc.reflectY = False
    freeParams = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    measured = gc.truePoints.copy()
    res = gc.loss(freeParams, gc.truePoints, measured)
    # how many points do we have
    exp = measured.shape[0] * 1.0
    assert abs(res) < exp + 0.1


def test_transform():
    """do not transform, image should stay the same"""
    orig_img = np.arange(16).reshape((4, 4))
    gc = GridCalibration_GridROI_COMExtractor(polynomialN=1)
    gc._R = np.eye(2)
    gc._T = np.array([0, 0])
    gc._S = 1.0 
    gc.reflectY = False
    # only linear in x on the x-and coord linear in y on the y coord
    # make sure to also shift, because pixels starts at 0, but metric system has the origin in the middle
    gc._params = np.array([-2.0, 0.0, 1.0, 0.0, +2.0, -1.0, 0.0, 0.0])
    # fake calib image of same size
    gc._calImgShape = orig_img.shape
    new_image = gc.transform(orig_img, outputResolution=(4, 4), outputSize=(4, 4))
    assert_array_almost_equal(new_image, orig_img)


def test_plot_img_and_rois():
    """three points, one in the center, one to the top, one to the right
    regions should be centered on those points"""
    orig_img = np.arange(25).reshape((5, 5))
    gc = GridCalibration_GridROI_COMExtractor(regularisation=False)
    # roi is in image coordinates (horizontal, vertical)
    rois = [
        RegionOfInterest(2, 2, (1, 1)),
        RegionOfInterest(2, -1, (1, 1)),
        RegionOfInterest(3, 2, (1, 1)),
    ]
    # same here, pixel coordinate system
    points = pd.DataFrame([(2,2), (2, -1), (3, 2)], columns=["x", "y"])
    gc.plot_img_and_rois(
        orig_img,
        rois,
        points,
        title="5x5, 3 points in center (2,2), right(3,2), and top(2,-1). Rois on points",
    )


def test_plot_img_and_rois_even():
    """put a 6x6 image with two rois and 3 measured points
    three points, one in the center, one to the top, one to the right
    regions should be centered on those points"""
    orig_img = np.arange(36).reshape((6, 6))
    gc = GridCalibration_GridROI_COMExtractor(regularisation=False)
    # roi is in image coordinates
    rois = [
        RegionOfInterest(2.5, 2.5, (1, 1)),
        RegionOfInterest(2.5, -0.5, (1, 1)),
        RegionOfInterest(3.5, 2.5, (1, 1)),
    ]
    points = pd.DataFrame([(2.5, 2.5), (2.5, -0.5), (3.5, 2.5)], columns=["x", "y"])
    gc.plot_img_and_rois(
        orig_img,
        rois,
        points,
        title="6x6, 3 points in center (2.5,2.5), right(3.5,2.5), and top(2.5,-0.5). Rois rounded to nearest int",
    )


"""
def test_on_calib_img_gridROI():
    #apply to real calibration image
    testfile = Path(r"./testfiles/genA_disCal_legacy.tif")
    calImg = np.asarray(Image.open(testfile))
    #optimised these values to agree pretty well with the image
    gc = GridCalibration_GridROI_COMExtractor(regularisation=False, offsetPxl=(35,-60), pxlSizeGuess=(0.22, 0.22))
    gc.fit(calImg)
    correctedImg = gc.transform(calImg)
    plt.title("Doesnt fit exactly, because of the distortion")
    plt.imshow(correctedImg)
    plt.show()
"""


def test_on_calib_img_HoughROI():
    """fit from real calibration image, use cartesian polynomials"""
    testfile = Path(r"./testfiles/genA_disCal_legacy.tif")
    calImg = np.asarray(Image.open(testfile))
    # optimised these values to agree pretty well with the image
    gc = GridCalibration_HoughROI_PassThrough_PointExtractor(
        polynomialN=1, regularisation=False
    )
    gc.debug = True
    gc.fit(calImg)
    correctedImg = gc.transform(
        calImg, outputResolution=calImg.T.shape, outputSize=(400, 400)
    )
    plt.title("Should now fit exactly")
    plt.imshow(correctedImg)
    plt.show()


def test_on_calib_img_HoughROI_shifted():
    """fit from real calibration image but shift it by 100 pixels, use cartesian polynomials"""
    testfile = Path(r"./testfiles/genA_disCal_legacy.tif")
    calImg = np.asarray(Image.open(testfile))
    calImg = np.roll(calImg, 140, 0)
    # optimised these values to agree pretty well with the image
    gc = GridCalibration_HoughROI_PassThrough_PointExtractor(
        polynomialN=1, regularisation=False
    )
    gc.debug = False
    gc.fit(calImg)
    correctedImg = gc.transform(
        calImg, outputResolution=calImg.T.shape, outputSize=(400, 400)
    )
    plt.title("Should now fit exactly")
    plt.imshow(correctedImg)
    plt.show()


def test_on_calib_img_HoughROI_3():
    """fit from real calibration image, higher polynomial"""
    testfile = Path(r"./testfiles/genA_disCal_legacy.tif")
    calImg = np.asarray(Image.open(testfile))
    # optimised these values to agree pretty well with the image
    gc = GridCalibration_HoughROI_PassThrough_PointExtractor(
        polynomialN=3, regularisation=False
    )
    gc.debug = False
    gc.fit(calImg)
    # should now fit exactly, still need to save the image output
    correctedImg = gc.transform(calImg, calImg.T.shape, (350, 350))
    plt.title("Should now fit exactly")
    plt.imshow(correctedImg)
    plt.show()


def test_manmatch_use_case():
    """load original calibration image and the distortion calibration and correct it
    This image always ended up off-center in manmatch, so lets make sure its not the distortion calib"""
    testfile = Path(r"./testfiles/cal_bs_000159.tif")
    calImg = np.asarray(Image.open(testfile))
    # optimised these values to agree pretty well with the image
    gc = GridCalibration_HoughROI_PassThrough_PointExtractor(
        polynomialN=3, regularisation=False
    )
    gc.low_threshold = 35
    gc.high_threshold = 80
    gc.debug = False
    gc.fit(calImg)
    print(f"Live result: {gc.write_to_string()}")
    # should now fit exactly, still need to save the image output
    correctedImg = gc.transform(calImg, calImg.T.shape, (350, 350))
    plt.title("Live result: Should now fit exactly")
    plt.imshow(correctedImg)
    plt.show()
    gc.read_from_string(
        "5.891753189830658,-0.04630804448376802,-4.534332932448862e-05,6.381402440165621e-07,1.0226241873893436,6.126218501602576e-05,-1.1323625701063127e-06,-1.7678660959224738e-09,-2.4875150068664718e-05,5.882933219022589e-07,5.620985428812031e-10,5.018202797849712e-12,-1.103553121406914e-06,-1.7702866449113435e-09,-1.700777203361137e-11,0.0,6.3930100220667745,1.0183002703115982,-2.59054645164867e-05,-1.099477208959863e-06,0.039510229395519195,1.0951254978674824e-05,-7.432286403608145e-07,-2.084454739046861e-09,-5.425464842863865e-05,-8.216400087181058e-07,-7.90444020085817e-11,-3.601257745027853e-11,-6.665657141597172e-07,-8.608594159840282e-10,0.0,0.0;0.2280595095889185;-191.88879743433668,180.67842930624195;0.9998818338506661,0.015372648939896526,-0.015372648939896514,0.9998818338506656"
    )
    correctedImg = gc.transform(calImg, calImg.T.shape, (350, 350))
    plt.title("DB result: Should now fit exactly")
    plt.imshow(correctedImg)
    plt.show()


"""
def test_on_calib_img_HoughROI_radial_3():
    testfile = Path(r"./testfiles/genA_disCal_legacy.tif")
    calImg = np.asarray(Image.open(testfile))
    #optimised these values to agree pretty well with the image
    gc = GridCalibrationRadial_HoughROI_PassThrough_PointExtractor(polynomialN=3, regularisation=False)
    gc.debug = False
    gc.fit(calImg)
    correctedImg = gc.transform(calImg, outputResolution=calImg.T.shape, outputSize=(400,400))
    plt.title("Should now fit exactly")
    plt.imshow(correctedImg)
    plt.show()

def test_on_calib_img_HoughROI_division():
    testfile = Path(r"./testfiles/genA_disCal_legacy.tif")
    calImg = np.asarray(Image.open(testfile))
    #optimised these values to agree pretty well with the image
    gc = GridCalibrationDivision_HoughROI_PassThrough_PointExtractor(regularisation=False)
    gc.debug = False
    gc.fit(calImg)
    correctedImg = gc.transform(calImg, outputResolution=calImg.T.shape, outputSize=(400,400))
    plt.title("Should now fit exactly")
    plt.imshow(correctedImg)
    plt.show()


def test_on_calib_img_HoughROI_division_logistic():
    testfile = Path(r"./testfiles/genA_disCal_legacy.tif")
    calImg = np.asarray(Image.open(testfile))
    #optimised these values to agree pretty well with the image
    gc = GridCalibrationDivisionLogistic_HoughROI_PassThrough_PointExtractor(regularisation=False)
    gc.debug = False
    gc.fit(calImg)
    correctedImg = gc.transform(calImg, outputResolution=calImg.T.shape, outputSize=(400,400))
    plt.title("Should now fit exactly")
    plt.imshow(correctedImg)
    plt.show()
"""


def test_repr():
    gc = GridCalibration_HoughROI_PassThrough_PointExtractor(
        polynomialN=1, regularisation=False
    )
    assert (
        repr(gc)
        == "GridCalibration_HoughROI_PassThrough_PointExtractor,polynomialN=1;regularisation=False;optimiser=Powell"
    )


def test_repr2():
    gc = GridCalibration_HoughROI_PassThrough_PointExtractor(
        polynomialN=3, regularisation=True
    )
    assert (
        repr(gc)
        == "GridCalibration_HoughROI_PassThrough_PointExtractor,polynomialN=3;regularisation=True;optimiser=Powell"
    )


def test_write_to_string():
    """must be able to write the fitted parameters to a string"""
    gc = GridCalibration_HoughROI_PassThrough_PointExtractor(
        polynomialN=1, regularisation=False
    )
    gc._params = np.array([1.1, 2.2, 3.3, 4.4])
    gc._S = 2.2
    gc._T = np.arange(2)
    gc._R = np.arange(4).reshape((2, 2))
    res = gc.write_to_string()
    exp = "1.1,2.2,3.3,4.4;2.2;0,1;0,1,2,3"
    assert res == exp


def test_read_from_string():
    """also read in parameters again from a string"""
    gc = GridCalibration_HoughROI_PassThrough_PointExtractor(
        polynomialN=1, regularisation=False
    )
    s = "1.1,2.2,3.3,4.4;2.2;0,1;0,1,2,3"
    gc.read_from_string(s)
    assert_array_almost_equal(gc._params, np.array([1.1, 2.2, 3.3, 4.4]))
    assert_array_almost_equal(gc._S, 2.2)
    assert_array_almost_equal(gc._R, np.arange(4).reshape((2, 2)))
    assert_array_almost_equal(gc._T, np.arange(2))
