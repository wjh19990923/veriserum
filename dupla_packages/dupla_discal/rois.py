""" Classes to define regions of interest

This is the first step in the distortion calibration process. Either automatically
or through manual intervention (e.g. selecting a certain bead) ROIs are defined and
are used to send parts of the image to the extract_point steps. Ideally only one bead
should be within each region of interest
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter, disk

from .synthetic_phantoms import SyntheticBeadGrid


class RegionOfInterest:
    def __init__(self, xPxl: int, yPxl: int, halfwidthPxl: tuple[int, int]):
        """xPxl, yPxl: midpoint of the ROI
        haldwidthPxl: x and y extent to the right and left, top and bottom of xPxl, yPxl
        ..note:: we convert input arguments to int (rounds down for positive numbers, up for negative)
        """
        if int(xPxl) != xPxl or int(yPxl) != yPxl:
            print("Warning: Converting input argument to int")
        self.xPxl = int(xPxl)
        self.yPxl = int(yPxl)
        self.halfwidthPxl = (int(halfwidthPxl[0]), int(halfwidthPxl[1]))

    @property
    def xMinPxl(self):
        return self.xPxl - self.halfwidthPxl[0]

    @property
    def xMaxPxl(self):
        return self.xPxl + self.halfwidthPxl[0]

    @property
    def yMinPxl(self):
        return self.yPxl - self.halfwidthPxl[1]

    @property
    def yMaxPxl(self):
        return self.yPxl + self.halfwidthPxl[1]

    def __repr__(self):
        return f"x: {self.xMinPxl}, {self.xMaxPxl}; y: {self.yMinPxl}, {self.yMaxPxl}"


class GridROIMixin:
    """cut the image into squares around the true bead positions, width=half inter-bead distance
    .. note:: doesnt yet consider stuff outside image range
    .. note:: assumes self.trueGrid"""

    def __init__(
        self, pxlSizeGuess=(400 / 1664, 400 / 1600), offsetPxl=(0, 0), *args, **kwargs
    ):
        """
        pxlSizeGuess
        offsetPxl - the offset in x and y direction, in pixels
        """
        self.pxlSizeGuess = pxlSizeGuess
        self.offsetPxl = offsetPxl
        super().__init__(*args, **kwargs)

    def find_regions_of_interest(self, img: np.array) -> list[RegionOfInterest]:
        # round up, so we have a larger ROI when in doubt
        halfwidthPxl = (
            np.ceil(self.trueGrid.sphereDistance[0] / (self.pxlSizeGuess[0] * 2)),
            np.ceil(self.trueGrid.sphereDistance[1] / (self.pxlSizeGuess[1] * 2)),
        )
        rois = []
        # calculate the x,y positions in pixels
        for x, y in zip(self.truePoints["x"], self.truePoints["y"]):
            x = x / self.pxlSizeGuess[0] + img.shape[1] // 2 + self.offsetPxl[0]
            y = -y / self.pxlSizeGuess[1] + img.shape[0] // 2 + self.offsetPxl[1]
            rois.append(RegionOfInterest(x, y, halfwidthPxl))
        return rois


class HoughROIMixin:
    """Use canny edge filter and hough transform to find the circular beads"""

    def __init__(self, *args, **kwargs):
        self.low_threshold = 25
        self.high_threshold = 100
        self.sigma = 3
        self.min_distance = 15  # minimal distance in pixels between circles
        self.halfwidthPxl = (20, 20)
        # we assume SIPLA legacy grid up to now
        grid = SyntheticBeadGrid(
            sphereDiameter=2,
            sphereNr=(45, 45),
            sphereDistance=(7, 7),
            offset=(0, 0),
            cutCornerWidth=11,
        )
        self.nrComponents = len(grid.beadPositions)
        super().__init__(*args, **kwargs)

    def find_regions_of_interest(
        self, img: np.array, debug=False
    ) -> list[RegionOfInterest]:
        edges = canny(
            img,
            sigma=self.sigma,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )
        if debug:
            plt.title("Detected edges")
            plt.imshow(edges)
            plt.show()

        # Detect the circles in the edge image
        hough_radii = np.arange(5, 10)
        hough_res = hough_circle(edges, hough_radii)

        # Select the most prominent circles
        accums, cx, cy, radii = hough_circle_peaks(
            hough_res,
            hough_radii,
            min_xdistance=self.min_distance,
            min_ydistance=self.min_distance,
            total_num_peaks=self.nrComponents,
        )

        image = color.gray2rgb(img)
        rois = []
        for center_y, center_x, radius in zip(cy, cx, radii):
            # circy, circx = circle_perimeter(center_y, center_x, radius,shape=image.shape)
            rr, cc = disk((center_y, center_x), radius)
            image[rr, cc] = (220, 20, 20)
            rois.append(RegionOfInterest(center_x, center_y, self.halfwidthPxl))
        if debug:
            # Draw them
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
            plt.title("Image and detected circles")
            ax.imshow(image, cmap=plt.cm.gray)
            plt.show()
        return rois
