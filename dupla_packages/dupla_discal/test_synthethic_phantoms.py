import matplotlib.pyplot as plt
from .synthetic_phantoms import SyntheticBeadGrid

"""
Remember that by default the intensifier is 400mm x 400mm
Renderer creates actually quite low differences in contrast, depending on sphereSize"""

# what do we call "dark"
THRESHOLD = 205
# show debug plots
DEBUG = False


def test_4_8_grid_1000pxl():
    """just a vanilla 1000x1000pxl grid, so 1pxl is 0.4mm, so the spheres are 10pxls from each other
    checking whether the center of the points is dark and some other points are white
    """
    grid = SyntheticBeadGrid(
        sphereDiameter=4,
        sphereNr=(4, 8),
        sphereDistance=(4, 4),
        offset=(0, 0),
        orientationSphere=False,
    )
    img = grid.render((1000, 1000))
    if DEBUG:
        plt.title(
            f"1000 pxl image, {grid.intensifierWidth}mm intensifier, 4mm intersphere distance, 4mm sphere diameter"
        )
        plt.imshow(img, cmap="gray")
        plt.colorbar()
        plt.show()
    assert img.shape == (1000, 1000)
    points = [(465 + 10 * row, 485 + 10 * col) for row in range(8) for col in range(4)]
    for row, col in points:
        assert img[row, col] < THRESHOLD, f"Value at {row,col} was {img[row,col]}"
    whitePoints = [(455, 485), (455, 495), (300, 300), (545, 505), (545, 495)]
    for row, col in whitePoints:
        assert img[row, col] == 255, f"Value at {row, col} was {img[row,col]}"


def test_2_1_grid():
    """different number of points, once again check that the points are dark, some other points are white"""
    grid = SyntheticBeadGrid(
        sphereDiameter=3,
        sphereNr=(2, 1),
        sphereDistance=(4, 4),
        offset=(0, 0),
        orientationSphere=False,
    )
    img = grid.render((1000, 1000))
    assert img.shape == (1000, 1000)
    points = [(500, 495), (500, 505)]
    for row, col in points:
        assert img[row, col] < THRESHOLD, f"Value at {row,col} was {img[row,col]}"
    whitePoints = [
        (505, 495),
        (495, 505),
        (455, 485),
        (455, 495),
        (300, 300),
        (545, 505),
        (545, 495),
    ]
    for row, col in whitePoints:
        assert img[row, col] == 255, f"Value at {row, col} was {img[row,col]}"


def test_1600pxl_1400pxl():
    """test another pixel output, remember that grid.render takes (W,H) but the shape is given in (H,W)"""
    grid = SyntheticBeadGrid(
        sphereDiameter=1,
        sphereNr=(4, 8),
        sphereDistance=(10, 10),
        offset=(0, 0),
        orientationSphere=False,
    )
    img = grid.render((1600, 1400))
    assert img.shape == (1400, 1600)


def test_offset_4():
    """offset by exactly one point to the right. Check that the unshifted pointsposition are white"""
    grid = SyntheticBeadGrid(
        sphereDiameter=3,
        sphereNr=(4, 8),
        sphereDistance=(4, 4),
        offset=(4, 0),
        orientationSphere=False,
    )
    img = grid.render((1000, 1000))
    assert img.shape == (1000, 1000)
    if DEBUG:
        plt.title("1000 shifted by 4mm == 10pxl")
        plt.imshow(img, cmap="gray")
        plt.colorbar()
        plt.show()
    points = [(465 + 10 * row, 495 + 10 * col) for row in range(8) for col in range(4)]
    for row, col in points:
        assert img[row, col] < THRESHOLD, f"Value at {row,col} was {img[row,col]}"
    whitePoints = [(455, 485), (455, 495), (300, 300), (545, 505), (545, 495)]
    # add the points that we shifted away from
    whitePoints += [(465 + 10 * row, 485) for row in range(8)]
    for row, col in whitePoints:
        assert img[row, col] == 255, f"Value at {row, col} was {img[row,col]}"


def test_3x3():
    """create a 3x3 grid"""
    grid = SyntheticBeadGrid(
        sphereDiameter=3,
        sphereNr=(3, 3),
        sphereDistance=(4, 4),
        offset=(0, 0),
        cutCornerWidth=0,
        orientationSphere=False,
    )
    img = grid.render((1000, 1000))
    assert img.shape == (1000, 1000)
    if DEBUG:
        plt.title("1000pxl, 3x3 beads")
        plt.imshow(img, cmap="gray")
        plt.colorbar()
        plt.show()
    points = [
        (500, 490),
        (500, 500),
        (500, 510),
        (490, 500),
        (510, 500),
        (490, 490),
        (490, 510),
        (510, 490),
        (510, 510),
    ]
    for row, col in points:
        assert img[row, col] < THRESHOLD, f"Value at {row,col} was {img[row,col]}"
    whitePoints = [(455, 485), (455, 495), (300, 300), (545, 505), (545, 495)]
    for row, col in whitePoints:
        assert img[row, col] == 255, f"Value at {row, col} was {img[row,col]}"


def test_3x3_corners():
    """create a 3x3 grid with 1 point in each corner cut"""
    grid = SyntheticBeadGrid(
        sphereDiameter=3,
        sphereNr=(3, 3),
        sphereDistance=(4, 4),
        offset=(0, 0),
        cutCornerWidth=1,
        orientationSphere=False,
    )
    img = grid.render((1000, 1000))
    if DEBUG:
        plt.title("1000pxl, 3x3 beads with corners")
        plt.imshow(img, cmap="gray")
        plt.colorbar()
        plt.show()
    assert img.shape == (1000, 1000)
    points = [(500, 490), (500, 500), (500, 510), (490, 500), (510, 500)]
    for row, col in points:
        assert img[row, col] < THRESHOLD, f"Value at {row,col} was {img[row,col]}"
    # cornerpoints
    whitePoints = [(490, 490), (490, 510), (510, 490), (510, 510)]
    # randompoints
    whitePoints += [(455, 485), (455, 495), (300, 300), (545, 505), (545, 495)]
    for row, col in whitePoints:
        assert img[row, col] == 255, f"Value at {row, col} was {img[row,col]}"


def test_8x4_corners():
    """create a 8x4 grid with 2 point in each corner cut"""
    grid = SyntheticBeadGrid(
        sphereDiameter=3,
        sphereNr=(8, 4),
        sphereDistance=(4, 4),
        offset=(0, 0),
        cutCornerWidth=2,
        orientationSphere=False,
    )
    img = grid.render((1000, 1000))
    assert img.shape == (1000, 1000)
    if DEBUG:
        plt.title("1000pxl, 8x4 beads with corners")
        plt.imshow(img, cmap="gray")
        plt.colorbar()
        plt.show()
    points = [(495, 495), (505, 505), (495, 475), (495, 525), (485, 495), (485, 505)]
    for row, col in points:
        assert img[row, col] < THRESHOLD, f"Value at {row,col} was {img[row,col]}"
    # cornerpoints
    whitePoints = [
        (465, 485),
        (475, 485),
        (525, 485),
        (535, 485),
        (465, 515),
        (475, 515),
    ]
    for row, col in whitePoints:
        assert img[row, col] == 255, f"Value at {row, col} was {img[row,col]}"
