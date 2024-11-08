"""
Originally we were trying to recreate legacy single-plane distortion calibration
./Projects/LMB_tools/Fluoroscopy/FlumoEvaluation/02 Programme/01 Entzerrung und Kamerakalibierung/Labview/FluoroCalibration/FluoroCalibration/
But it basically uses Ni_VIsion_Development:IMAQ Learn Calibration Template, so nobody has an idea what it does

instead we develop our own methods

We are reducing the use of pandas in parts of the code because of performance issues and to use numba

.. note::
    DeformableRegistration, RigidRegistration, and minimize need proper tolerances to know when to stop, affecting
    performance and speed considerably
"""
from scipy import optimize, interpolate
import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
from pycpd import DeformableRegistration, RigidRegistration
from numba import njit

from .synthetic_phantoms import SyntheticBeadGrid
from .rois import RegionOfInterest, GridROIMixin, HoughROIMixin
from .extract_points import CenterOfMass_PointExtractor, PassThrough_PointExtractor


def disCal_by_db_method(method):
    """Returns the correct disCal class by giving the method in the database `distortion_calibration.method`"""
    if method == "whatever":
        return GridCalibration_HoughROI_PassThrough_PointExtractor(
            polynomialN=3, regularisation=False
        )
    else:
        raise ValueError(f"Unknown discalMethod {method}")


def metric_to_pixel_coordinates(
        x, y, outputSize: tuple[int, int], outputResolution: tuple[int, int]
):
    """
    convert the pos(mm) to pixels, assuming that we have a picture spanning
    outputSize in mm space, and is centered around 0mm,0mm
    outputResolution: W,H of output (full image), e.g. (1664,1600)
    outputSize: W,H dimensions in mm of the output image, e.g. (400,400)
    invert: remember that y axis goes up in mm-space and down in image space
    """
    newX = x * outputResolution[0] / outputSize[0] + outputResolution[0] / 2
    y = (-1) * y
    newY = y * outputResolution[1] / outputSize[1] + outputResolution[1] / 2
    return newX, newY


@njit
def calc_polynomial(
        freeParams,
        powers,
        points,
        out,
        S=0.9,
        R=np.eye(2),
        T=np.array([0.0, 0.0]),
        reflectY=True,
) -> np.ndarray:
    """use jit, powers is now an array with each row being the power of x and the power y
    fit both x and y with a 3rd order polynomial, actually see GridCalibration. Because we want to use numba, we need to make this a separate function
        Args:
            freeParams - (32,1), tuple of free parameters, first 16 belong to x
            powers - (polynomialNr + 1, 2), holds the coefficient [i,j] for the term x^i y^j
            points - (nsamples, 2) with columns "x" and "y" the coordinates of the measured points
            out - just a placeholder array of size points.shape with given dtype
            S, R, T - scale, rotation matrix, translation vector for the rigid transformation
            reflectY - if true, first multiply y by (-1)
        Returns:
            (nsamples, 2)
        .. note:: powers in x,y are ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0),
        (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3))
        .. note::
    """
    if reflectY:
        points[:, 1] *= -1
    # apply the rigid transformation
    for n in range(points.shape[0]):
        # dumb doing matrix multiplication likes this
        out[n, 0] = S * (R[0, 0] * points[n, 0] + R[0, 1] * points[n, 1]) + T[0]
        out[n, 1] = S * (R[1, 0] * points[n, 0] + R[1, 1] * points[n, 1]) + T[1]
    for n in range(points.shape[0]):
        temp = np.zeros(powers.shape)
        for p in range(powers.shape[0]):
            temp[p, 0] = (
                    freeParams[p]
                    * pow(out[n, 0], powers[p, 0])
                    * pow(out[n, 1], powers[p, 1])
            )
            temp[p, 1] = (
                    freeParams[len(freeParams) // 2 + p]
                    * pow(out[n, 0], powers[p, 0])
                    * pow(out[n, 1], powers[p, 1])
            )
        out[n, 0] = np.sum(temp[:, 0])
        out[n, 1] = np.sum(temp[:, 1])
    return out


class BaseCalibration:
    """
    Base class for image transformers follows roughly the design of the sklearn library
    All estimators and transformers should specify all the parameters \
    that can be set at the class level in their ``__init__`` as \
    explicit keyword arguments (no ``*args`` or ``**kwargs``).
    """

    def __init__(self):
        """debug - set to True if you want more plots"""
        self.truePoints = pd.DataFrame({"x": (0, 0, 1, 1), "y": (0, 1, 0, 1)})
        self.initialGuess = (0, 1)  # gives also the dimensionality of freeParams
        self._params = ()
        self.debug = False

    def fit(self, calibImgs: np.ndarray):
        """set parameters from information in the measurement
        save them in self._params
        no return (should not change anything)
        """
        pass

    def forward(self, points: np.ndarray, *args, **kwargs) -> np.ndarray:
        """apply fitting function to points using self._params, which is found by running self.fit()
        Args:
            points - (nsamples, 2) the x and y coordinates of the points
        Returns:
            (nsamples,2) - transformed points
        """
        return self.fitting_function(self._params, points, *args, **kwargs)

    def fitting_function(self, freeParams: tuple[float], points: pd.DataFrame):
        """fitting function for points with coordinates (x,y)
        Args:
            freeParams - (nParams,1), tuple of free parameters
            points - (nsamples,2) the x and y coordinates of the points
        Returns:
            (nsamples, 2)
        """
        pass

    def fitting_function_noPandas(self, freeParams: tuple[float], pointsX, pointsY):
        """a pandas free variant"""
        pass

    def loss(self, freeParams: tuple[float], measuredPoints):
        """loss function to optimise between calibrated measured geometry and true geometry"""
        pass

    def extract_points(self, img):
        """find coordinates of calibration geometry (e.g. grid) in the image
        Implemented by a PointExtractor in extract_points.py"""
        pass

    def transform(
            self,
            img: np.ndarray,
            outputResolution: tuple[float:float],
            outputSize: tuple[float, float],
    ) -> (np.ndarray):
        """Uses self._params to transform sample
        resamples to size
        outputResolution: of output (full image), e.g. (1664,1600)
        outputSize: the dimensions in mm of the output image, e.g. (400,400)
        """
        pass

    def write_to_string(self):
        """write self._params in string form for database storage"""
        pass

    def read_from_string(self, s: str):
        """read self._params from database string"""
        pass

    def __repr__(self):
        """representation that contains all the necessary information to be written to the database"""
        pass


class GridCalibration(BaseCalibration):
    """calibration for the legacy grid phantom
    distance between the beads is 7mm, 45*45 beads

    1) regions of interest are around each true bead
    2) find the coordinate of each ROI by calculate the center of mass
    3) perform a rigid registration to find a scale, translation and rotation
    4) fit a 3rd order polynomial in (x,y)


    .. note:: Example for n = 3, powers in x,y are ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0),
    (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3))
    """

    def __init__(self, polynomialN=3, regularisation: float = False):
        """
        Args:
            polynomialN: the polynomial order to fit (in both x and y, cross terms are double order)
            regularisation: if float use L2 regularisation with the number as weight

        .. note:: self.rigidTolerance and self.deformableTolerance matter in terms of runtime
        """
        super().__init__()
        # create the true points for the legacy grid
        self.trueGrid = SyntheticBeadGrid(
            sphereDiameter=2,
            sphereNr=(45, 45),
            sphereDistance=(7, 7),
            offset=(0, 0),
            cutCornerWidth=11,
        )
        self.truePoints = self.trueGrid.beadPositions
        # highest order of x, and highest order of y
        self.polynomialN = polynomialN
        self.regularisation = regularisation
        # holds the optimised coefficients as np.array
        self._params = None
        # tolerance for the deformable registrations for establishing correspondences, scale -1 to +1
        self.deformableTolerance = 1e-5
        # this happens on a mm scale
        self.rigidTolerance = 1e-2
        # max iterations for rigid/deformable registrations, usually doesnt matter
        self.maxIterations = 200
        # rigid registration rotation, translation, scaling to be applied before the calibration function, just to make minimisation easier
        self._R = np.eye(2)
        self._T = np.array([-800 / 4, +800 / 4])
        self._S = 0.25
        # the reflection that we always apply when going from pixel coordinates (y going up) to metric coordinates (y going down)
        self.reflectY = True
        # hold the resolution of the calibration image, set during `self.fit()`, used for safety checks
        self._calImgShape = (1600, 1664)
        # the method for np.optimize.minimize
        self.optimiserMethod = "Powell"

    @property
    def y_offset(self):
        """offset for the y coefficients in the coefficient array (half)
        .. example:: for 3rd order polynomials, we have 16 coefficients for x, then 16 coefficients for y"""
        return (self.polynomialN + 1) * (self.polynomialN + 1)

    @property
    def pixel_sizes(self) -> dict[float, float]:
        """average pixel sizes in mm for horizontal and vertical
        transform the topleft (0,0) pxl and the bottom-right pxl given by saved calibration image resolution
        and divide the difference by the resolution"""
        if self._params is not None and self._calImgShape is not None:
            df = pd.DataFrame(
                {"x": [0, self._calImgShape[1]], "y": [0, self._calImgShape[0]]}
            )
            new = self.forward(df)
            h = (new.x.max() - new.x.min()) / self._calImgShape[1]
            v = (new.y.max() - new.y.min()) / self._calImgShape[0]
            return {"h": abs(h), "v": abs(v)}

    def give_rois(self, calibImgs: np.ndarray):
        """find points in the image, and run the optimisation against the loss function and truePoints
                results are saved in self._params
                .. note:: rigid registration used to be performed before running the minimsation. but just using the standard values
                """
        logging.info("Extracting points from image ...")
        self._calImgShape = calibImgs.shape
        rois = self.find_regions_of_interest(calibImgs, debug=self.debug)
        self.rois = rois
        return rois

    def fit(self, calibImgs: np.ndarray):
        """find points in the image, and run the optimisation against the loss function and truePoints
        results are saved in self._params
        .. note:: rigid registration used to be performed before running the minimsation. but just using the standard values
        """
        logging.info("Extracting points from image ...")
        self._calImgShape = calibImgs.shape
        rois = self.find_regions_of_interest(calibImgs, debug=self.debug)
        self.rois = rois
        measuredGeometry = self.extract_points(calibImgs, rois)
        measuredGeometry = self.sort_measured(measuredGeometry)
        if self.debug:
            self.plot_img_and_rois(
                calibImgs, rois, measuredGeometry, "Sorted ROIs and measured points"
            )
        startT = time.time()
        # for the rigid registration we need to implement the reflection manually
        foo = measuredGeometry.copy()
        foo.y *= -1
        """
        #easier to just use the fixed initial values
        reg = RigidRegistration(
            **{"X": self.truePoints.values, "Y": foo.values},
            max_iterations=10,
            scale=True
        )
        reg.register()
        self._R = reg.R
        self._T = reg.t
        self._S = reg.s
        """
        if self.debug:
            # plot true points and predicted points from initial guess
            self._params = self.initial_guess(calibImgs, measuredGeometry)
            predictions = self.forward(measuredGeometry)
            loss = self.loss(self._params, self.truePoints, measuredGeometry)
            self.scatter_pairwise(
                self.truePoints,
                predictions,
                f"True points and predictions using the intial guess (incl. rigid registration). Loss: {loss}",
            )
        logging.info(
            f"Optimisation started with polynomials of orders {self.polynomialN}, optimiser: {self.optimiserMethod} "
            f"and regularisation: {self.regularisation}"
        )
        res = optimize.minimize(
            self.loss,
            x0=self.initial_guess(
                calibImgs, measuredGeometry
            ),  # just set linear coefficients to 1
            args=(self.truePoints, measuredGeometry),
            method=self.optimiserMethod,
            options={"disp": True, "maxiter": self.maxIterations},
        )
        endT = time.time()
        logging.info(f"Optimisation finished after {endT - startT}")
        if res.success:
            self._params = res.x
            logging.info(f"Parameters: {self._params} \n Loss: {res.fun}")
            if self.debug:
                predictions = self.forward(measuredGeometry)
                # plot true points and predicted points (from extracted points + optimised params)
                self.scatter_pairwise(
                    self.truePoints,
                    predictions,
                    f"Predicted points (x), and true points (o), overall loss: {res.fun}",
                )
                # also show a plot of the errors
                euclidian = np.sqrt(
                    (predictions.x - self.truePoints.x) ** 2
                    + (predictions.y - self.truePoints.y) ** 2
                )
                euclidian.plot(kind="box")
                plt.title(
                    "Euclidian distances between predictions and true points (mm)"
                )
                plt.show()
        else:
            raise RuntimeError(f"Minimization failed with {res}")

    def initial_guess(self, calibImgs, measuredGeometry) -> np.ndarray:
        """initial guess: find the max and min of each axis, and adjust bias and linear coefficients
        Example of order for polynomial=2: ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2))
        """
        initialGuess = np.zeros(2 * self.y_offset)
        # linear coefficient x
        initialGuess[self.polynomialN + 1] = 1.0
        # linear coefficient y
        initialGuess[self.y_offset + 1] = 1.0
        return initialGuess

    def fitting_function(
            self, freeParams: tuple[float], points: np.ndarray
    ) -> np.ndarray:
        """fit both x and y with a 3rd order polynomial
        Args:
            freeParams - (32,1), tuple of free parameters, first 16 belong to x
            points - (nsamples, 2) with columns x and y the coordinates of the measured points
        Returns:
            (nsamples, 2) - colum 0 is x, and column 1 is y
        .. note:: powers in x,y are ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0),
        (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3))
        .. note::
            self._R, self._S, self._T - rigid transformation to be applied before
            self.reflectY, whether y axis is reflected
        """
        if isinstance(points, pd.DataFrame):
            p = points.values
        else:
            p = points
        assert freeParams.shape == (
            2 * self.y_offset,
        ), f"Expected {2 * self.y_offset} free parameters, got {freeParams.shape}"
        # create the powers for the polynomial as tuple, a generator would get used up by next comprehension
        powers = np.array(
            tuple(
                (i, j)
                for i in range(self.polynomialN + 1)
                for j in range(self.polynomialN + 1)
            )
        )
        out = calc_polynomial(
            freeParams,
            powers,
            p.astype(np.float64),
            np.zeros(p.shape, dtype=np.float64),
            S=self._S,
            R=self._R,
            T=self._T,
            reflectY=self.reflectY,
        )
        return pd.DataFrame(out, columns=["x", "y"])

    def loss(self, freeParams: tuple[float], truePoints: pd.DataFrame, measuredPoints: pd.DataFrame):
        """loss function to optimise between calibrated measured geometry and true geometry
        measuredPoints: in pixels (origin in the center of the image)
        log of sum of square loss with L2 regularisation of all but bias coefficient (if self.regularisation)
        self.regularisation is used as a weight (if not None, False or 0)
        """
        pred = self.fitting_function(freeParams, measuredPoints)
        # can just substract x from x and y from y
        res = (pred - truePoints) ** 2
        # add regularisation
        if self.regularisation:
            res += self.regularisation * np.sum(freeParams[1: self.y_offset] ** 2)
            res += self.regularisation * np.sum(freeParams[self.y_offset + 1:] ** 2)
        res = res.to_numpy().sum()
        return np.log(res)

    def sort_measured(self, measuredGeometry: pd.DataFrame):
        """some extractors return the measuredPoints in a random order (e.g. when using HoughROI)
        we now need to find correspondences before we can find the calibration function
        1. normalize to approximately align center of clouds and make their scale comparable
        2. then use CPD to find correspondences
        Args:
            measuredGeometry: colums x and y in pixel coordinates"""

        def normalize_col(col):
            return (col - col.mean()) / abs((col - col.mean()).max())

        reference = self.truePoints.apply(normalize_col)
        measured = measuredGeometry.apply(normalize_col)
        # as this function doesnt use forward (and reflectY) we need to manually flip the y axis to do rigid registration
        measured.y *= -1
        reg = DeformableRegistration(
            **{
                "X": reference.values,
                "Y": measured.values,
                "max_iterations": self.maxIterations,
                "tolerance": self.deformableTolerance,
            }
        )
        reg.register()
        # find the most likely correspondences
        indices = np.argmax(reg.P, axis=1).flatten()
        if self.debug:
            aligned_points = reg.TY
            aligned_points = pd.DataFrame(aligned_points, columns=["x", "y"])
            aligned_points["indices"] = indices
            aligned_points.sort_values("indices", inplace=True)
            aligned_points.reset_index(inplace=True)
            del aligned_points["index"]
            del aligned_points["indices"]
            self.scatter_pairwise(
                reference,
                aligned_points,
                title="Normalised true points and aligned points",
            )
        assert len(set(indices)) == len(
            indices
        ), "Double indices found, so we have a problem finding correspondences"
        # now sort the original measured points to correspond to the true points and return
        measuredGeometry["indices"] = indices
        sortedGeo = measuredGeometry.sort_values("indices")
        sortedGeo.reset_index(inplace=True)
        del sortedGeo["index"]
        del sortedGeo["indices"]
        return sortedGeo

    @staticmethod
    def scatter_pairwise(df1, df2, title):
        """plot pairs of points from two dataframes in the same color
        need both to have x and y columns
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        assert df1.shape == df2.shape, "Dataframes must have the same shape"
        colors = cm.flag(np.linspace(0, 1, df1.shape[0]))
        for i in range(df1.shape[0]):
            ax.scatter(df1.iloc[i].x, df1.iloc[i].y, marker="o", color=colors[i])
            ax.scatter(df2.iloc[i].x, df2.iloc[i].y, marker="x", color=colors[i])
        plt.title(title)
        plt.show()

    def plot_img_and_rois(
            self,
            img: np.ndarray,
            rois: list[RegionOfInterest],
            points: list[tuple[float, float]],
            title: str = None,
    ):
        """plot img and regions of interest, as well as extracted points
        Args:
            img: the image
            rois: list of RegionOfInterest
            points: the extracted points (in pxl), pixel coordinate system
            title: optional plot title
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap="gray")
        for roi in rois:
            ax.add_patch(
                Rectangle(
                    (roi.xMinPxl, roi.yMinPxl),
                    2 * roi.halfwidthPxl[0],
                    2 * roi.halfwidthPxl[1],
                    lw=1,
                    facecolor="none",
                    ec="g",
                )
            )
        ax.scatter(points.x, points.y)
        if title:
            plt.title(title)
        else:
            plt.title("Measurement and ROIs")
        plt.show()

    def transform(
            self,
            img: np.array,
            outputResolution: tuple[float:float],
            outputSize: tuple[float, float],
    ) -> np.array:
        """Correct original image using the calibrated function
        Args:
            img - input image, will be left unchanged
            outputResolution: W,H of output (full image), e.g. (1600,1600), most of the time this should be square
            outputSize: W,H dimensions in mm of the output image, e.g. (400,400)
        Returns:
            newImg - the transformed image
        ..note:: fill_value of interpolation is 0.
        ..note:: We only support input images of the same resolution as the calibration image. This is not necessary
        but seems like a sensible check for the time being
        """
        assert (
                img.shape == self._calImgShape
        ), f"Image has a different resolution {img.shape} than the calibration image {self._calImgShape}"
        # for each pixel, find the new position. Remember that origin is in the middle
        x = np.arange(img.shape[1])
        y = np.arange(img.shape[0])
        assert (
                   len(y),
                   len(x),
               ) == img.shape, f"Image shape was {img.shape} but new positions had shape {(len(y), len(x))}"
        xx, yy = np.meshgrid(x, y)
        df = np.vstack([xx.flatten(), yy.flatten()]).T
        assert df.shape == (img.shape[1] * img.shape[0], 2)
        # get the new positions of each pixel according to calibration function (so we got from pxl to mm)
        newPos = self.forward(df)
        # convert back to pixels, flip y, because it still is in mm order where y goes up
        pixelNewX, pixelNewY = metric_to_pixel_coordinates(
            newPos["x"], newPos["y"], outputSize, outputResolution
        )
        # get pixels in new resolution
        pixelX = np.arange(outputResolution[0])
        pixelY = np.arange(outputResolution[1])
        px, py = np.meshgrid(pixelX, pixelY)
        newImg = interpolate.griddata(
            (pixelNewY, pixelNewX),
            img.flatten(),
            (py, px),
            method="nearest",
            fill_value=0.0,
        )
        assert newImg.shape == (
            outputResolution[1],
            outputResolution[0],
        ), f"Transposed new image.shape {newImg.T.shape} did not match output resolution {outputResolution[1], outputResolution[0]}. Programming bug"
        if self.debug:
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(img, cmap="gray")
            ax[0].set_title("Original image")
            ax[1].imshow(newImg, cmap="gray")
            pixelTrueX, pixelTrueY = metric_to_pixel_coordinates(
                self.truePoints.x, self.truePoints.y, outputSize, outputResolution
            )
            ax[1].scatter(pixelTrueX, pixelTrueY)
            ax[1].set_title("Transformed image and true points")
            plt.show()
        return newImg

    def write_to_string(self):
        """write self._params in string form for database storage
        just returns a comma separated list, parts separated by ;"""
        ps = ",".join([str(n) for n in self._params])
        s = str(self._S)
        t = ",".join([str(n) for n in self._T.flatten()])
        r = ",".join([str(n) for n in self._R.flatten()])
        return ps + ";" + s + ";" + t + ";" + r

    def read_from_string(self, s: str):
        """read self._params from database string"""
        parts = s.split(";")
        self._params = np.array([float(num) for num in parts[0].split(",")])
        self._S = float(parts[1])
        self._T = np.array([float(num) for num in parts[2].split(",")])
        self._R = np.array([float(num) for num in parts[3].split(",")]).reshape((2, 2))


class GridCalibrationRadial(GridCalibration):
    """..note:: might be fine, but not supported atm"""

    """same as GridCalibration but with a radial polynomial
    doesnt correct overall shift"""

    def fitting_function(self, freeParams: tuple[float], points: pd.DataFrame):
        """fit both x and y with one polynomial
        Args:
            freeParams - (self.polynomialN+1,), tuple of free parameters (0th order, r, r^2, r^3)
            points - (nsamples, 2) with columns "x" and "y" the coordinates of the measured points
        Returns:
            (nsamples, 2)
        .. note::
            self._R, self._S, self._T - rigid transformation to be applied before
        """
        assert freeParams.shape == (
            self.polynomialN + 1,
        ), f"Freeparams was expected to have shape {(self.polynomialN + 1,)} but was found to have {freeParams.shape}"
        points = points.apply(
            lambda row: pd.Series(self._S * self._R @ row + self._T, index=["x", "y"]),
            axis="columns",
            result_type="expand",
        )
        points["r"] = np.sqrt(points["x"] ** 2 + points["y"] ** 2)
        points["phi"] = np.arctan2(points["y"], points["x"])
        powers = [i for i in range(self.polynomialN + 1)]
        r = sum(freeParams[n] * pow(points.r, i) for n, i in enumerate(powers))
        xCalc = r * np.cos(points.phi)
        yCalc = r * np.sin(points.phi)
        # when naming it x and y we can directly subtract the corresponding true columns in loss()
        return pd.DataFrame({"x": xCalc, "y": yCalc})

    def initial_guess(self, calibImgs, measuredGeometry) -> np.ndarray:
        """initial guess: just set linear coefficient to 1"""
        initialGuess = np.zeros((self.polynomialN + 1))
        initialGuess[1] = 1.0
        return initialGuess


class GridCalibrationDivision(GridCalibration):
    """..note:: might be fine, but not supported atm"""

    """use the division model https://en.wikipedia.org/wiki/Distortion_(optics) """

    def fitting_function(self, freeParams: tuple[float], points: pd.DataFrame):
        """fit both x and y with one polynomial
         Args:
             freeParams - (xc, yc, k1, k2), same for x and y
             points - (nsamples, 2) with columns "x" and "y" the coordinates of the measured points
         Returns:
             (nsamples, 2)
         .. note::
             self._R, self._S, self._T - rigid transformation to be applied before
         .. note:: if you get rid of the fourth order term loss goes from 5200 to 5500
             typical result would be: [-3.82337433e+00,  2.29435245e+00, -1.85645834e-03,  1.31288348e-05,
        -1.30587462e-10] loss: 5245
        """
        assert freeParams.shape == (
            5,
        ), f"Freeparams was expected to have shape (4,) but was found to have {freeParams.shape}"
        xc, yc, k1, k2, k4 = freeParams
        points = points.apply(
            lambda row: pd.Series(self._S * self._R @ row + self._T, index=["x", "y"]),
            axis="columns",
            result_type="expand",
        )
        points["r"] = np.sqrt(points["x"] ** 2 + points["y"] ** 2)
        points["phi"] = np.arctan2(points["y"], points["x"])
        xCalc = xc + (points.x - xc) / (
                1 + k1 * points.r + k2 * points.r ** 2 + k4 * points.r ** 4
        )
        yCalc = yc + (points.y - yc) / (
                1 + k1 * points.r + k2 * points.r ** 2 + k4 * points.r ** 4
        )
        # when naming it x and y we can directly subtract the corresponding true columns in loss()
        return pd.DataFrame({"x": xCalc, "y": yCalc})

    def initial_guess(self, calibImgs, measuredGeometry) -> np.ndarray:
        initialGuess = np.zeros(5)
        return initialGuess


class GridCalibrationDivisionLogistic(GridCalibration):
    """..note:: might be fine, but not supported atm"""

    def fitting_function(self, freeParams: tuple[float], points: pd.DataFrame):
        """fit both x and y with one polynomial in radial direction (division model) and add a sigmoid function for the x coordinates
        the sigmoid changes amplitude with x, but is centered on the x-axis
        Args:
            freeParams - (xc, yc, k1, k2, a0, a1), same for x and y
            points - (nsamples, 2) with columns "x" and "y" the coordinates of the measured points
        Returns:
            (nsamples, 2)
        .. note::
            self._R, self._S, self._T - rigid transformation to be applied before

        use the division model https://en.wikipedia.org/wiki/Distortion_(optics)
        """
        assert freeParams.shape == (
            8,
        ), f"Freeparams was expected to have shape (6,) but was found to have {freeParams.shape}"
        xc, yc, k1, k2, k4, a0, a1, cx = freeParams
        points = points.apply(
            lambda row: pd.Series(self._S * self._R @ row + self._T, index=["x", "y"]),
            axis="columns",
            result_type="expand",
        )
        points["r"] = np.sqrt(points["x"] ** 2 + points["y"] ** 2)
        points["phi"] = np.arctan2(points["y"], points["x"])
        xCalc = (
                xc
                + (points.x - xc)
                / (1 + k1 * points.r + k2 * points.r ** 2 + k4 * points.r ** 4)
                + (a0 + 1 / a1 * points.x) / (1 + np.exp(-cx * (points.y - 0)))
        )
        yCalc = yc + (points.y - yc) / (
                1 + k1 * points.r + k2 * points.r ** 2 + k4 * points.r ** 4
        )
        return pd.DataFrame({"x": xCalc, "y": yCalc})

    def initial_guess(self, calibImgs, measuredGeometry) -> np.ndarray:
        """
        can be used, but we use an experience based value
        initialGuess = np.zeros(8)
        initialGuess[6] = 100
        """
        initialGuess = np.array(
            2.32010409e01,
            1.21414649e00,
            -6.46494794e-04,
            -1.11454680e-06,
            1.46749715e-10,
            3.01126820e00,
            1.20916684e02,
            -3.18959412e-02,
        )

        return initialGuess


"""###########################################
######    Combination of ROI and Calibration
###########################################"""


class GridCalibration_GridROI_COMExtractor(
    GridROIMixin, CenterOfMass_PointExtractor, GridCalibration
):
    """
    ROI: GridSplit
    ExtractPoints: center of mass
    Fitting: powers of x up to 3 and y up to 3, highest order is x3y3
    Optimiser: Powell, can be set through self.optimiserMethod
    Loss: L2 loss and L2 regularisation of all but bias coefficients, give the coefficient
    """

    def __init__(
            self,
            pxlSizeGuess=(400 / 1660, 400 / 1600),
            offsetPxl=(0.0, 0.0),
            polynomialN=3,
            regularisation=None,
    ):
        super().__init__(
            pxlSizeGuess=pxlSizeGuess,
            offsetPxl=offsetPxl,
            polynomialN=polynomialN,
            regularisation=regularisation,
        )


class GridCalibration_HoughROI_COMExtractor(
    HoughROIMixin, CenterOfMass_PointExtractor, GridCalibration
):
    """
    ROI: HoughROI
    Fitting: powers of x up to 3 and y up to 3, highest order is x3y3
    Optimiser: Powell, can be set through self.optimiserMethod
    Loss: L2 loss and L2 regularisation of all but bias coefficients, give the coefficient
    """

    def __init__(self, polynomialN=3, regularisation=False):
        super().__init__(polynomialN=polynomialN, regularisation=regularisation)


class GridCalibration_HoughROI_PassThrough_PointExtractor(
    HoughROIMixin, PassThrough_PointExtractor, GridCalibration
):
    """
    ROI: HoughROI
    Extractor: just use ROIs from Hough
    Fitting: powers of x up to 3 and y up to 3, highest order is x3y3
    Optimiser: Powell, can be set through self.optimiserMethod
    Loss: L2 loss and L2 regularisation of all but bias coefficients, give the coefficient
    """

    def __init__(self, polynomialN=3, regularisation=False):
        super().__init__(polynomialN=polynomialN, regularisation=regularisation)

    def __repr__(self):
        return f"{self.__class__.__name__},polynomialN={self.polynomialN};regularisation={self.regularisation};optimiser={self.optimiserMethod}"


class GridCalibrationRadial_HoughROI_PassThrough_PointExtractor(
    HoughROIMixin, PassThrough_PointExtractor, GridCalibrationRadial
):
    """
    ROI: HoughROI
    Extractor: just use ROIs from Hough
    Fitting: radial fitting up to polynomialN, same for x and y
    Optimiser: Powell, can be set through self.optimiserMethod
    Loss: L2 loss and L2 regularisation of all but bias coefficients, give the coefficient
    """

    def __init__(self, polynomialN=3, regularisation=False):
        super().__init__(polynomialN=polynomialN, regularisation=regularisation)

    def __repr__(self):
        return f"{self.__class__},polynomialN={self.polynomialN};regularisation={self.regularisation};optimiser={self.optimiserMethod}"


class GridCalibrationDivision_HoughROI_PassThrough_PointExtractor(
    HoughROIMixin, PassThrough_PointExtractor, GridCalibrationDivision
):
    """
    ROI: HoughROI
    Extractor: just use ROIs from Hough
    Fitting: radial fitting using the division model (order 0, 2, 4),same for x and y
    Optimiser: can be set through self.optimiserMethod
    Loss: L2 loss and L2 regularisation of all but bias coefficients, give the coefficient
    """

    def __init__(self, regularisation=False):
        super().__init__(regularisation=regularisation)

    def __repr__(self):
        return f"{self.__class__},regularisation={self.regularisation};optimiser={self.optimiserMethod}"


class GridCalibrationDivisionLogistic_HoughROI_PassThrough_PointExtractor(
    HoughROIMixin, PassThrough_PointExtractor, GridCalibrationDivisionLogistic
):
    """
    ROI: HoughROI
    Extractor: just use ROIs from Hough
    Fitting: radial fitting using the division model (order 0, 2, 4),same for x and y, and add a logistic function for the x coordinate
    Optimiser: can be set through self.optimiserMethod
    Loss: L2 loss and L2 regularisation of all but bias coefficients, give the coefficient
    """

    def __init__(self, regularisation=False):
        super().__init__(regularisation=regularisation)

    def __repr__(self):
        return f"{self.__class__},regularisation={self.regularisation};optimiser={self.optimiserMethod}"


if __name__ == "__main__":
    import cProfile as profiler
    import pstats

    testfile = Path(r"dupla_discal/testfiles/cal_bs_000159.tif")
    calImg = np.asarray(Image.open(testfile))
    # optimised these values to agree pretty well with the image
    gc = GridCalibration_HoughROI_PassThrough_PointExtractor(
        polynomialN=3, regularisation=False
    )
    gc._calImgShape = calImg.shape
    gc.debug = False
    gc.read_from_string(
        "5.891753189830658,-0.04630804448376802,-4.534332932448862e-05,6.381402440165621e-07,1.0226241873893436,6.126218501602576e-05,-1.1323625701063127e-06,-1.7678660959224738e-09,-2.4875150068664718e-05,5.882933219022589e-07,5.620985428812031e-10,5.018202797849712e-12,-1.103553121406914e-06,-1.7702866449113435e-09,-1.700777203361137e-11,0.0,6.3930100220667745,1.0183002703115982,-2.59054645164867e-05,-1.099477208959863e-06,0.039510229395519195,1.0951254978674824e-05,-7.432286403608145e-07,-2.084454739046861e-09,-5.425464842863865e-05,-8.216400087181058e-07,-7.90444020085817e-11,-3.601257745027853e-11,-6.665657141597172e-07,-8.608594159840282e-10,0.0,0.0;0.2280595095889185;-191.88879743433668,180.67842930624195;0.9998818338506661,0.015372648939896526,-0.015372648939896514,0.9998818338506656"
    )
    profiler.runctx(
        "[gc.transform(calImg, calImg.T.shape, (350,350)) for i in range(10)]",
        globals(),
        locals(),
        sort="tottime",
    )
