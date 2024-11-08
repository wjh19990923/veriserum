"""Defines the physical properties (size, geometry) of the phantoms used for distortion calibration
Can create render images of these phantoms, or can create `Anatomies` that can be placed in another `Scene`
"""
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from PIL import Image
import pandas as pd

# from dupla_renderers.OpenGL.opengl_renderer import GLRenderer
# from dupla_renderers.OpenGL.scene_tools import Scene, Camera, Disk, ModelBase


class SyntheticBeadGrid:
    def __init__(
        self,
        sphereDiameter: float,
        sphereNr: tuple[int, int],
        sphereDistance: tuple[float, float],
        offset: tuple[float, float],
        cutCornerWidth: int = 0,
        orientationSphere=True,
    ):
        """Radio-opaque beads in a rectilinear grid
        Args:
            sphereDiameter: in mm
            sphereNr: how many sphere in horizontal and vertical direction
            sphereDistance: distance in mm between spheres in hor and vert direction
            offset: (horizontal, vertical), in mm. (0,0) means that the origin is put in the center of the grid by default)
            cutCornerWidth: cut off the corners like in the SIPLA grid phantom, max width of one triangle
            orientationSphere: the sipla grid has one additional sphere for orientation above left top diagonal

        .. note:: origin is in the middle of the grid
        .. note:: resolution of the texture on the image intensifier sets the intensifier size (through pixelSize)
        and the field-of-view (we only look at the intensifier)
        resolution of the renderer givess the resolution of the rendered image
        """
        # doesnt matter, because we always look at the complete intensifier
        self.focalLength = 1000
        self.intensifierWidth = self.intensifierHeight = 400
        self.centerPos = np.array((0, 0, 0))
        self.normalVec = np.array((0, 0, 1))
        self.verticalVec = np.array((0, 1, 0))
        self.sphereDiameter = sphereDiameter
        self.sphereNr = sphereNr
        self.sphereDistance = sphereDistance
        self.offset = offset
        self.cutCornerWidth = cutCornerWidth

        """beadIdx is something like
            [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), 
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), 
            (2, 0), ... if we dont cut corners"""
        beadIdx = list(self.beadTuples(sphereNr, cutCornerWidth))
        # add the orientation sphere above the top left diagonal
        if orientationSphere:
            beadIdx.append((5, sphereNr[1] - 6))
        beadPositions = [
            (
                sphereDistance[0] * (x - (sphereNr[0] - 1) / 2) + offset[0],
                sphereDistance[1] * (y - (sphereNr[1] - 1) / 2) + offset[1],
            )
            for x, y in beadIdx
        ]
        self.beadPositions = pd.DataFrame(beadPositions, columns=("x", "y"))

    def create_models(self):
        """create `Sphere` anatomies at the `self.beadPositions` with the size `self.sphereDiameter`
        can be placed into a Scene for rendering"""
        actors = []
        for x, y in zip(self.beadPositions.x, self.beadPositions.y):
            a = Disk(name=x + y, diameter=self.sphereDiameter, nfrags=30)
            # this messes things up because it transfers to the next loop or something
            # tmat = a.get_model_matrix()
            tmat = np.eye(4)
            tmat[0, 3] = x
            tmat[1, 3] = y
            tmat[2, 3] = 0
            a.apply_new_model_matrix(tmat)
            actors.append(a)
        return actors

    def render(self, outputResolution: tuple[int, int]):
        """render image
        Args:
            outputResolution: (width, height) in pixels
        Returns:
            img - shape outputResolution, datatype is np.uint8
        """
        renderer = GLRenderer()
        renderer.set_resolution(*outputResolution)
        # create spheres at each bead position
        actors = self.create_models()
        cam = Camera(
            name="cam",
            screen_world_pos=self.centerPos,
            screen_normal=self.normalVec,
            screen_vert=self.verticalVec,
            screen_size_width=self.intensifierWidth,
            screen_size_height=self.intensifierHeight,
            principal_point_h=0,
            principal_point_v=0,
            focal_length=self.focalLength,
        )
        scene = Scene(actors=actors, cameras=[cam])
        renderer.bind_scene(scene)
        rendered_img = renderer.render(0)
        # just take the red channel
        return rendered_img[:, :, 0]

    def beadTuples(self, imgShape: tuple[int, int], xWidth: int):
        """matrix with everything 1 except corner triangles, tuple of elements that are 1
            xWidth - number of elements to set to zero in the first/last row, decreased by one each row
        Returns:
            Tuple of elements equal to 1, in matrix (row,col) order
        .. example::
            00011000
            00111100
            01111110
            11111111
            11111111
            01111110
            00111100
            00011000
        gives (0,3), (0,4), (1,2), (1,3), ...
        """
        if xWidth == 0:
            return ((x, y) for x in range(imgShape[0]) for y in range(imgShape[1]))
        else:
            indices = np.zeros(imgShape)
            for row in range(xWidth):
                if xWidth - row >= 0:
                    indices[row, xWidth - row : -(xWidth - row)] = 1
                    indices[-row - 1, xWidth - row : -(xWidth - row)] = 1
            indices[(xWidth):-xWidth, :] = 1
            w = np.where(indices == 1)
            return ((x, y) for x, y in zip(*w))
