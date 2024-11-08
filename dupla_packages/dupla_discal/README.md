This is the repository for distortion calibation/correction

In the end, all methods take an input image, 
1. find regions of interest in the image. E.g. finding a window around each bead. Not required for global extraction methods
2. extract a geometric pattern, either globally or from each region of interest
3. compare this measured pattern with an ideal pattern and optimise a function to minimize the error 
4. transform another image by using this calibration function (interpolation)

Extracting geometry for the legacy grid phantom:
Each beads is a solid sphere and thus the ideal radiographic projection
should just be a half-circle in 1D or a halfsphere in 2D.
In our images, this is complicated by a) the rasterization (diameter of the circle is around 8pixels), b) imaging noise, 
c) the discretization of values (likely to be irrelevant compared to the other effects).

# ROI extractors
1. create ROIs from regular grid. This is problematic if there is too much distortion or too much rotation
2. use Hough transform to detect circles

# Extract geometry pattern
1. Theoretically one could model the underlying ideal density, then apply rasterization and noise, and try to find a maximum-likelihood solution (or maximum posterior probability). This is not trivial.
2. As a simpler approach, we calculate the center of mass
3. Alternatively, one could directly use the centers from the hough-transform used for ROI detection



Restructure:
Calibration: needs to implement pack and unpack
