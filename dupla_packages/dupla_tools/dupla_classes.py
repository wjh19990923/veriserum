"""Container classes to go from the more general tools in the `dupla_renderer` to the concrete application \
of the DUPLA system with 2 planes and 2 anatomies. Includes convenience functions to hold all information of 
a single frame, to work with calibrations and poses, also defines default values

This file uses the OpenGL functions of the renderers, for the p3d version, see dupla_classes_p3d
"""
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
import numpy as np
from matplotlib.pyplot import get_cmap
from scipy.spatial.transform import Rotation
from skimage import io
from ..dupla_discal import GridCalibration_HoughROI_PassThrough_PointExtractor
from ..dupla_discal.synthetic_phantoms import SyntheticBeadGrid
# from dupla_renderers.OpenGL.scene_tools import (
#     Anatomy,
#     Camera,
#     Intensifier,
#     Scene,
#     create_intensifier_camera,
# )
# from dupla_sical.synthetic_phantoms import SiCalPhantom


def img_from_path(img_path) -> np.ndarray:
    """load image from image path as 16 bit image"""
    startTime = datetime.now()
    img = io.imread(img_path).astype(dtype=np.uint16, casting="unsafe")
    endTime = datetime.now()
    logging.info(f"Loading image {img_path} took {endTime-startTime}")
    assert img.ndim == 2, "image must be single channel grayscale."
    logging.info(f"Loaded image statistics: min={img.min()}, max={img.max()}, dtype={img.dtype}")
    return img


def parse_measurementIdx_string(configStr: str) -> list[int]:
    """parse the config measurements string of the form 1,2,3-5,7
    Example:
        1,2,3-5,7,11 return [1,2,3,4,5,7,11]"""
    ids = []
    for s in configStr.split(","):
        s = s.strip()
        if "-" in s:
            start, stop = s.split("-")
            ids += list(range(int(start.strip()), int(stop.strip()) + 1, 1))
        else:
            ids.append(int(s))
    return ids



class InterIntensifierCalibration:
    def __init__(self):
        pass

    def load_default_values(self):
        self.bs_pos = np.array((0.0, 0, 0))
        self.bs_normal_vec = np.array((0.0, 0, 1))
        self.bs_vertical_vec = np.array((0.0, 1, 0))
        self.fs_pos = np.array((360.0, 0, 250))
        self.fs_vertical_vec = np.array((0.0, 1, 0))
        self.fs_normal_vec = np.dot(
            Rotation.from_rotvec(np.deg2rad(110 - 180) * np.array([0, 1, 0])).as_matrix(),
            self.bs_normal_vec,
        )

    def load_from_dict(self, d):
        self.bs_pos = d["bs_pos"]
        self.bs_normal_vec = d["bs_normal_vec"]
        self.bs_vertical_vec = d["bs_vertical_vec"]
        self.fs_pos = d["fs_pos"]
        self.fs_vertical_vec = d["fs_vertical_vec"]
        self.fs_normal_vec = d["fs_normal_vec"]


class SourceIntensifierCalibration:
    def __init__(self):
        # the source_int_calibration.id
        self.idx = None
        self.calibrationTrialIdxFs = None
        self.calibrationTrialIdxBs = None

    def load_default_values(self):
        self.bs_pph = 0.0
        self.bs_ppv = 0.0
        self.bs_sid = 1350  # in mm
        self.fs_pph = 0.0
        self.fs_ppv = 0.0
        self.fs_sid = 1350  # in mm

    def load_from_dict(self, d):
        """load calibration_trial_ids, principal points and SIDs from dictionary"""
        self.idx = d["id"]
        self.calibrationTrialIdxFs = d["calibration_trial_id_fs"]
        self.calibrationTrialIdxBs = d["calibration_trial_id_bs"]
        self.fs_pph = d["principalp_h_fs"]
        self.fs_ppv = d["principalp_v_fs"]
        self.bs_pph = d["principalp_h_bs"]
        self.bs_ppv = d["principalp_v_bs"]
        self.fs_sid = d["sid_fs"]
        self.bs_sid = d["sid_bs"]

    def __eq__(self, other):
        return (
            self.bs_pph == other.bs_pph
            and self.bs_ppv == other.bs_ppv
            and self.bs_sid == other.bs_sid
            and self.fs_pph == other.fs_pph
            and self.fs_ppv == other.fs_ppv
            and self.fs_sid == other.fs_sid
        )

    def __repr__(self):
        return f"SourceIntensifierCalibration: {self.__dict__}"


class DistortionCalibration:
    def __init__(self):
        # normal camera resolution of dupla, which we use for the calibration images
        # image to be transformed needs to be the same resolution as precaution
        self.CALSHAPE = (1600, 1664)
        self.calibrationTrialIdxFs = None
        self.calibrationTrialIdxBs = None
        # distortion_calibration.id
        self.idx = None

    def load_from_dict(self, d):
        """
        must include calibration_trial_id, distortion_result_bs, distortion_result_fs, and method
        """
        if d["method"] == "whatever":
            self.idx = d["id"]
            self.calibrationTrialIdxFs = d["calibration_trial_id_fs"]
            self.calibrationTrialIdxBs = d["calibration_trial_id_bs"]
            self.fs_dc = GridCalibration_HoughROI_PassThrough_PointExtractor(
                polynomialN=3, regularisation=False
            )
            self.fs_dc.read_from_string(d["distortion_result_fs"])
            self.fs_dc._calImgShape = self.CALSHAPE
            self.bs_dc = GridCalibration_HoughROI_PassThrough_PointExtractor(
                polynomialN=3, regularisation=False
            )
            self.bs_dc.read_from_string(d["distortion_result_bs"])
            self.bs_dc._calImgShape = self.CALSHAPE

    def load_default_values(self):
        """just set to None and transform is going to pass through"""
        self.fs_dc = None
        self.bs_dc = None

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return f"DistortionCalibration: {self.__dict__}"

    def transform_fs(self, img, outputResolution: int, outputSize: float):
        """outputSize in mm"""
        if self.fs_dc is None:
            return img
        else:
            startTime = datetime.now()
            newImg = self.fs_dc.transform(img, (outputResolution, outputResolution), (outputSize, outputSize))
            endTime = datetime.now()
            logging.info(f"Transforming FS took {endTime-startTime}")
            return newImg

    def transform_bs(self, img, outputResolution: int, outputSize: float):
        """outputSize in mm"""
        if self.bs_dc is None:
            return img
        else:
            startTime = datetime.now()
            newImg = self.bs_dc.transform(img, (outputResolution, outputResolution), (outputSize, outputSize))
            endTime = datetime.now()
            logging.info(f"Transforming BS took {endTime-startTime}")
            return newImg


class Pose:
    def __init__(
        self,
        anatomy_idx,
        pose_idx: int = None,
        source_intensifier_calibration_id: int = None,
        distortion_calibration_id: int = None,
    ):
        """corresponding to the database fields in `poses`"""
        self.idx = pose_idx
        self.siCal_idx = source_intensifier_calibration_id
        self.disCal_idx = distortion_calibration_id
        self.anatomy_idx = anatomy_idx
        self.tmat = None

    def load_default_values(self):
        tmat = np.eye(4)
        tmat[0:3, 3] = [0, 0, 500]  # tx, ty, tz
        self.tmat = tmat

    def load_quaternion(
        self,
        tx: float,
        ty: float,
        tz: float,
        r0: float,
        r1: float,
        r2: float,
        r3: float,
    ):
        tmat = np.eye(4)
        tmat[0:3, 3] = (tx, ty, tz)
        tmat[0:3, 0:3] = Rotation.from_quat([r0, r1, r2, r3]).as_matrix()
        self.tmat = tmat

    def to_ortho6d(self):
        """returns 9 components representation, 3 translation and the first two rows of the rotation matrix"""
        return np.hstack((self.tmat[0:3, 3], self.tmat[0:3, 0:2].transpose().flatten()))

    def to_dict(self):
        """return a dictionary with tx, ty, tz, r0, r1, r2, r3"""
        tx, ty, tz = self.tmat[0:3, 3]
        r0, r1, r2, r3 = Rotation.from_matrix(self.tmat[0:3, 0:3]).as_quat()
        return {"tx": tx, "ty": ty, "tz": tz, "r0": r0, "r1": r1, "r2": r2, "r3": r3}

    def __eq__(self, other):
        return self.tmat == other.tmat

    def __repr__(self):
        return f"Pose with tmat: {self.tmat}"

    def copy(self):
        p = Pose(anatomy_idx = self.anatomy_idx,
                   pose_idx = self.idx,
                   source_intensifier_calibration_id = self.siCal_idx,
                   distortion_calibration_id = self.disCal_idx)
        p.tmat = self.tmat.copy()
        return p






class SiplaDistortionGrid(SyntheticBeadGrid):
    """the geometry of the single plane distortion grid made of lead beads in plexiglas"""

    def __init__(self):
        super().__init__(
            sphereDiameter=2,
            sphereNr=(45, 45),
            sphereDistance=(7, 7),
            offset=(0, 0),
            cutCornerWidth=11,
        )
