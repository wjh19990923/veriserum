""" Classes to extract points or find a certain geometry in an image or a part
of an image corresponding to a ROI

This is the second step in the distortion calibration pipeline. For each ROI
we try to detect the geometry, and return the coordinates. Later, this will be used
to fit a function by comparing these predictions to the true geometry of the phantom

All of these classes are supposed to be mixins that implement the extract_points() method

coordinate system: origin at top left, x going right, y going down
"""
import numpy as np
import pandas as pd
from .rois import RegionOfInterest


class PassThrough_PointExtractor:
    """just return the center of the regions of interest
    .. note:: useful, if you want the ROI detector also as point extractor (e.g. HoughROIMixin)
    .. warning:: ROI extractors usually return integer, because they are centered on pixels
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_points(self, img, regions: RegionOfInterest) -> pd.DataFrame:
        """return center of the regions of interest
        Args:
            regions: Pass Regions of Interest
        Returns:
            DataFrame with columns x,y: in pixels rounded to 0.1 pixels
        """
        points = [(roi.xPxl, roi.yPxl) for roi in regions]
        return pd.DataFrame(points, columns=("x", "y"))


class CenterOfMass_PointExtractor:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def calculate_center_of_mass(img: np.array, decimals=1) -> tuple[float, float]:
        """find the center of mass
        each pixel votes for its center position with a weight proportional to its value
        center values start at 0. (so the leftmost pixel actually extends to -0.5)
        Args:
            decimals - round to this number of decimals
        Returns:
            x,y coordinate of the COM. In pixels from the top-left.
        """
        # (n x m) * (mxm)
        m = np.zeros((img.shape[1], img.shape[1]))
        np.fill_diagonal(m, np.arange(img.shape[1]))
        xCoords = np.ones(img.shape) @ m
        # (n x n) (n x m)
        n = np.zeros((img.shape[0], img.shape[0]))
        np.fill_diagonal(n, np.arange(img.shape[0]))
        yCoords = n @ np.ones(img.shape)
        xCOM = (xCoords * img).sum() / img.sum()
        yCOM = (yCoords * img).sum() / img.sum()
        return xCOM, yCOM

    def extract_points(self, img, regions: list[RegionOfInterest]) -> pd.DataFrame:
        """find coordinates of calibration geometry (e.g. grid) in the image
        Args:
            regions: Pass Regions of Interest
        Returns:
            DataFrame with columns x,y: in pixels rounded to 0.1 pixels
        """
        # find the center of mass within each region
        points = list()
        for roi in regions:
            subImg = img[roi.yMinPxl : roi.yMaxPxl, roi.xMinPxl : roi.xMaxPxl]
            assert subImg.shape == (
                roi.yMaxPxl - roi.yMinPxl,
                roi.xMaxPxl - roi.xMinPxl,
            ), f"Subimage shape {subImg.shape} with ROI {roi}"
            # calculate the COM within the subimage, from top left of subimage
            x, y = self.calculate_center_of_mass(subImg)
            # add the general position of the ROI, and shift the overall origin to the center of the image
            x = x + roi.xMinPxl
            y = y + roi.yMaxPxl
            points.append((x, y))
        return pd.DataFrame(points, columns=("x", "y"))
