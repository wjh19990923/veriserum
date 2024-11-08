import logging
from datetime import datetime
from pathlib import Path
from torch import Tensor

import numpy as np
from matplotlib.pyplot import get_cmap
from scipy.spatial.transform import Rotation

from dupla_renderers.pytorch3d import AnatomyCT, Scene, Camera
from dupla_renderers.pytorch3d.pytorch3d_renderer import CTRenderer
from dupla_renderers.pytorch3d.utilities import kneefit_to_pytorch3d_renderer

from .rotation_tools import ortho6d_from_rotation_matrix, rotation_matrix_from_ortho6d, tmat_from_9d

class CachedCTRenderer():
    """implement a cache for the anatomies"""
    def __init__(self):
        self._cachedTibiaPath = None
        self._cachedFemurPath = None
        self._cachedTibia = None
        self._cachedFemur = None

    def create_xray(
        self, femurPath: Path, tibiaPath: Path, calDict: dict, pose: np.array,
        dataset, device="cpu", N=1, reverse_transform=False
    ) -> np.array:
        """
        Args:
            pose: [batchsize, 18], can be None, then we use the default poses
            femurPath, tibiaPath: 3d volumetric data, one can be None
            dataset: the underlying dataset
            N: only render the first N samples
            reverse_transform: run reverse_transform
        Obsolete:
            reverseTransform: to apply to pose, must leave pose in original units at (1, 18)
                determined by  dataset.reverse_transform
            outsizePxl: size of the rendering, determined by dataset.outSizePxl
        Returns:
            imgs, [batchsize, camerNr, S, S] rendered images

        .. note:: because kneefit uses different conventions, we need to convert some
            things to use it with pytorch3d
        .. warning:: reloads data when it encounters new anatomies
        """
        assert device, "Please pass a device"
        requiredCals = ("cal_mm_per_pxl", "cal_principalp_x", "cal_principalp_y", "cal_focal_length")
        assert set(requiredCals).issubset(set(calDict.keys())), "Missing required info"
        if femurPath is None:
            self._cachedFemurPath = None
            self._cachedFemur = None
        elif not (femurPath == self._cachedFemurPath):
            self._cachedFemurPath = femurPath
            self._cachedFemur = AnatomyCT.load_data(femurPath, multiply=N)
        femur = self._cachedFemur

        if tibiaPath is None:
            self._cachedTibiaPath = None
            self._cachedTibia = None
        elif not (tibiaPath == self._cachedTibiaPath):
            self._cachedTibiaPath = tibiaPath
            self._cachedTibia = AnatomyCT.load_data(tibiaPath, multiply=N)
        tibia = self._cachedTibia

        renderer = CTRenderer(device=device)
        scene = Scene()
        assert femur or tibia, "Please pass either femur or tibia (or both)"
        if femur:
            femur = femur.to(device)
            scene.add_anatomies(femur)
        if tibia:
            tibia = tibia.to(device)
            scene.add_anatomies(tibia)
        if not isinstance(pose, Tensor):
            pose = Tensor(pose).to(device)
        rCalDict = dict()
        #get the required cal infos and convert to Tensor if necessary
        for k in requiredCals:
            v = calDict[k]
            if isinstance(v, Tensor):
                rCalDict[k] = v[:N].to(device)
            else:
                rCalDict[k] = Tensor(v[:N], device=device)
        batchsize = femur._N if femur else tibia._N
        cameras = Camera(
            [f"camera_{n}" for n in range(batchsize)],
            screen_center_poses=Tensor((0, 0, 0)).expand(N, 3).to(device),
            screen_normals=Tensor((0, 0, 1)).expand(N, 3).to(device),
            screen_verticals=Tensor((0, 1, 0)).expand(N, 3).to(device),
            screen_sizes_h=1000 * rCalDict["cal_mm_per_pxl"].to(device),
            screen_sizes_v=1000 * rCalDict["cal_mm_per_pxl"].to(device),
            principal_points_h=rCalDict["cal_principalp_x"].to(device),
            principal_points_v=rCalDict["cal_principalp_y"].to(device),
            focal_lengths=rCalDict["cal_focal_length"].to(device),
        )
        scene.add_cameras(cameras)
        renderer.bind_scene(scene)
        if not (pose is None):
            if reverse_transform:
                logging.disable(logging.WARNING)
                _, pose, _ = dataset.transformer.reverse_transform(None, pose, None)
                logging.disable(logging.NOTSET)
            # pose is back (18,) in original units
            assert pose.shape[-1] == 18
            if femur:
                femurTMats = tmat_from_9d(pose[:N, :9]).to(device)
                femurTMats, femurWorldT = kneefit_to_pytorch3d_renderer(
                    femur.original_intrinsic_affine_mat(), femurTMats, rCalDict
                )
            if tibia:
                tibiaTMats = tmat_from_9d(pose[:N, 9:]).to(device)
                tibiaTMats, tibiaWorldT = kneefit_to_pytorch3d_renderer(
                    tibia.original_intrinsic_affine_mat(), tibiaTMats, rCalDict
                )
            # render both
            if femur and tibia:
                assert femur._N == tibia._N
                renderer.scene.anatomies[0].set_model_matrix(
                    femurTMats, is_yours=False, theirs_to_yours=femurWorldT
                )
                renderer.scene.anatomies[1].set_model_matrix(
                    tibiaTMats, is_yours=False, theirs_to_yours=tibiaWorldT
                )
            # only a tibia is passed
            elif tibia:
                renderer.scene.anatomies[0].set_model_matrix(
                    tibiaTMats, is_yours=False, theirs_to_yours=tibiaWorldT
                )
            # only a femur
            else:
                renderer.scene.anatomies[0].set_model_matrix(
                    femurTMats, is_yours=False, theirs_to_yours=femurWorldT
                )
        imgs = renderer.render(
            cam_index=0,
            width_pixels_num=dataset.outSizePxl,
            height_pixels_num=dataset.outSizePxl,
            binary=False,
            scale_output=True,
            efficient=True,
        )
        return imgs

