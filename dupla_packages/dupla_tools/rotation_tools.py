"""offers tools to convert between different rotational representations and adjust poses

the reason we are implementing these is that the np implementations would not use
torch variables and therefore have no autograd
much of the original code can be accredited to:
https://github.com/papagina/RotationContinuity/blob/master/shapenet/code/tools.py
and
On the Continuity of Rotation Representations in Neural Networks - Zhou

In general we use the convention of rotating the vector, so 90deg around z means rotating the vector

.. note:: refactoring in progress, moved the file from damn_healthy_knee to here
"""
import numpy as np
import torch


def deg2rad(d):
    return d * np.pi / 180


def rad2deg(r):
    return r * 180 / np.pi

def tmat_from_9d(pose):
    """convert 9d representation to 4x4 matrix"""
    assert pose.shape[1] == 9
    poseMat = torch.zeros((pose.shape[0], 4, 4)).to(pose.device)
    poseMat[:, 0:3, 0:3] = rotation_matrix_from_ortho6d(pose[:, 3:9])
    poseMat[:, 0:3, 3] = pose[:, :3]
    poseMat[:, 3, 3] = 1.
    return poseMat

def adjust_pose(pose, change):
    """convert 9d representation to 4x4 matrix, and multiply from right with change
    Args:
        pose9d: (B, 9)
        change: (4,4)
    Returns:
        (B, 9)
    """
    assert change.shape == (4,4)
    assert pose.shape[1] == 9
    poseMat = mat_from_9d(pose) @ change.to(pose.device)
    new_pose = torch.zeros((pose.shape[0], 9)).to(pose.device)
    new_pose[:, 0:3] = poseMat[:, 0:3, 3]
    new_pose[:, 3:6] = poseMat[:, 0:3, 0]
    new_pose[:, 6:9] = poseMat[:, 0:3, 1]
    return new_pose

def create_random_tmat(pm):
    """
    create a random 3x4 matrix with translations +/- pm mm
    """
    tmat = torch.eye(4)
    tmat[0:3, 3] = (torch.rand(3)-0.5) * pm
    return tmat



# euler batch*4
def rotation_matrix_from_euler(rx: torch.tensor, ry: torch.tensor, rz: torch.tensor) -> torch.tensor:
    r"""calculate rotation matrix from Flumatch Euler angles z-> x'-> y''(intrinsic)

    Args:
    rx, ry, rz: (batch_nr * 1), in *degrees*
      dtype of the tensor must be supported by torch.sin and torch.cos (e.g. not LongTensor)

    Returns:
    mat: (batch_nr x 3 x 3) rotation matrices

    \left(
    \begin{array}{ccc}
     \cos (\text{ry}) \cos (\text{rz})-\sin (\text{rx}) \sin (\text{ry}) \sin (\text{rz}) & -\cos (\text{rx}) \sin (\text{rz}) & \cos (\text{rz}) \sin (\text{ry})+\cos (\text{ry}) \sin (\text{rx}) \sin (\text{rz}) \\
     \cos (\text{rz}) \sin (\text{rx}) \sin (\text{ry})+\cos (\text{ry}) \sin (\text{rz}) & \cos (\text{rx}) \cos (\text{rz}) & \sin (\text{ry}) \sin (\text{rz})-\cos (\text{ry}) \cos (\text{rz}) \sin (\text{rx}) \\
     -\cos (\text{rx}) \sin (\text{ry}) & \sin (\text{rx}) & \cos (\text{rx}) \cos (\text{ry}) \\
    \end{array}
    \right)

    also see: https://www.geometrictools.com/Documentation/EulerAngles.pdf

    Note: Euler angles mean rotating the vector, so rz=90 means that (1, 0, 0) goes into (0,1,0)
    """
    assert rx.shape == ry.shape == rz.shape, "Please pass tensors of equal lengths"
    # passed tensor shapes of (batch_nr,) instead of (batch_nr, 1)
    if len(rx.shape) == 1:
        rx = torch.unsqueeze(rx, 1)
        ry = torch.unsqueeze(ry, 1)
        rz = torch.unsqueeze(rz, 1)
    # convert to radian
    rx = deg2rad(rx)
    ry = deg2rad(ry)
    rz = deg2rad(rz)
    # dim: batch_nr * 1
    s_rx = torch.sin(rx).unsqueeze(1)
    s_ry = torch.sin(ry).unsqueeze(1)
    s_rz = torch.sin(rz).unsqueeze(1)
    c_rx = torch.cos(rx).unsqueeze(1)
    c_ry = torch.cos(ry).unsqueeze(1)
    c_rz = torch.cos(rz).unsqueeze(1)
    # batch_nr*1*3
    row1 = torch.cat(
        (
            c_ry * c_rz - s_rx * s_ry * s_rz,
            -c_rx * s_rz,
            c_rz * s_ry + c_ry * s_rx * s_rz,
        ),
        2,
    )
    row2 = torch.cat(
        (
            c_rz * s_rx * s_ry + c_ry * s_rz,
            c_rx * c_rz,
            -c_ry * c_rz * s_rx + s_ry * s_rz,
        ),
        2,
    )
    row3 = torch.cat((-c_rx * s_ry, s_rx, c_rx * c_ry), 2)
    # batch_nr*3*3
    matrix = torch.cat((row1, row2, row3), 1)
    return matrix


def euler_from_rotation_matrix(
    m: torch.tensor,
) -> (torch.tensor, torch.tensor, torch.tensor):
    """inverse of 'rotation_matrix_from_euler', returns angles in *degrees*
    check there for details on conventions and the underlying rotation matrix
    Args:
        m: shape is (batch_nr, 3, 3)

    see
    https://www.geometrictools.com/Documentation/EulerAngles.pdf

    need to change the quadrants and signs to correspond the the flumatch convention of angles/quadrants

    Returns:
      rx, ry, rz - torch.tensors of form (batch_nr, 1), in *degrees*
    """
    rx = torch.asin(m[:, 2, 1])  # the sin(rx) term
    ry = -torch.atan2(m[:, 2, 0], m[:, 2, 2])  # -cos(rx)sin(ry) and cos(rx) cos(ry)
    rz = -torch.atan2(m[:, 0, 1], m[:, 1, 1])  # -cos(rx)sin(rz) and cos(rx) cos(rz)

    rx = rad2deg(rx)
    ry = rad2deg(ry)
    rz = rad2deg(rz)

    return rx.view(-1, 1), ry.view(-1, 1), rz.view(-1, 1)


def orthogonalize_ortho6d(ortho6d):
    """normalize all components, and make sure they are orthogonal, see
    `rotation_matrix_from_ortho6d'"""
    mat = rotation_matrix_from_ortho6d(ortho6d)
    return ortho6d_from_rotation_matrix(mat)

def rotation_matrix_from_ortho6d(ortho6d):
    """for SO(3) the 6d representation is just the 3x3 rotation matrix
    with the last column thrown out

    so we need to restore the last column by a cross product (to ensure orthogonality to other
    columns)
    then we get column 2 from the crossproduct, to ensure it is orthogonal to both 1 and 3
    (as 1 and 2 didnt need to be orthogonal, but should have been)

    ..note:: normalise Frobenius norm / 2-norm is used, according to the defaults of `torch.norm`

    Args:
      ortho6d: (batch_nr, 6)

    Returns:
      matrix: shape (batch_nr, 3,3)
    """
    x_raw = ortho6d[:, 0:3]  # first column, batch*3
    y_raw = ortho6d[:, 3:6]  # second column, batch*3
    x = x_raw / torch.norm(x_raw, dim=1, keepdim=True)  # batch*3
    z_raw = torch.cross(x, y_raw, dim=1)  # batch*3
    z = z_raw / torch.norm(z_raw, dim=1, keepdim=True)  # batch*3
    y = torch.cross(z, x, dim=1)  # batch*3
    y = y / torch.norm(y, dim=1, keepdim=True)  # just to ensure its numerically 1

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def ortho6d_from_rotation_matrix(m):
    """for SO(3) the 6d representation is just the 3x3 rotation matrix
    with the last column thrown out

    Args:
      m - the rotation matrix
    Returns:
      matrices with one column less, so shape (batch_nr, 3, 2)
    """
    assert m.shape[1] == 3 and m.shape[2] == 3, "Only 3x3 matrices are supported"
    r = m[:, :, :-1]  # get rid of last column
    r = r.permute(0, 2, 1)  # swap rows of the rotation matrix and columns
    r = r.reshape((m.shape[0], -1))  # so flatten gives the right order
    return r


def ortho6d_from_euler(rx: torch.tensor, ry: torch.tensor, rz: torch.tensor) -> torch.tensor:
    """
    Args:
      rx,ry, rz: shape(batch_nr, 1)
    """
    m = rotation_matrix_from_euler(rx, ry, rz)
    return ortho6d_from_rotation_matrix(m)


def euler_from_ortho6d(ortho6d):
    """
    Args:
      ortho6d: (batch_nr, n) with n typically 9

    Returns:
      rx, ry, rz - torch.tensors of form (batch_nr, 1), in *degrees*
    """
    m = rotation_matrix_from_ortho6d(ortho6d)
    return euler_from_rotation_matrix(m)

