import numpy as np
import numpy.matlib
from scipy.spatial.transform import Rotation as Rot
import tensorflow as tf
# import cv2
import time
from tqdm import tqdm
try:
    from tf3d.utils.voxel_utils import pointcloud_to_sparse_voxel_grid
except:
    print("Cannot load tf3d")
import math
from sklearn.neighbors import NearestNeighbors


def get_bounding_box(P):
    """Get the bounding box of a point cloud P.

    Parameters
    ----------
    P : np.ndarray
        Point cloud.

    Returns
    -------
    tuple(float, float, float, float, float, float)
        Bounding box.

    """
    return (
            np.min(P[:, 0]),
            np.max(P[:, 0]),
            np.min(P[:, 1]),
            np.max(P[:, 1]),
            np.min(P[:, 2]),
            np.max(P[:, 2]))


def get_general_bb(P):
    bb = np.zeros((2*P.shape[1], ))
    j = 0
    for i in range(P.shape[1]):
        bb[j] = np.min(P[:, i])
        j += 1
        bb[j] = np.max(P[:, i])
        j += 1
    return bb


def get_rotation_mat(angle, axis):
    """Returns the rotation matrix from an angle and axis.

    Parameters
    ----------
    angle : float
        Angle of the rotation.
    axis : int
        Axis (x, y, z) of the rotation.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.

    """
    r = Rot.from_rotvec(angle * axis)
    return r.as_matrix()


def gl_perspective(angle_of_view, aspect_ratio, n):
    """Set up the variables that are necessary to compute the perspective
    projection matrix. See also:
    https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml

    Parameters
    ----------
    angle_of_view : float
         Specifies the field of view angle, in degrees, in the y direction.
    aspect_ratio : float
        Specifies the aspect ratio that determines the field of view in the x
        direction. The aspect ratio is the ratio of x (width) to y (height).
    n : float
        Specifies the distance from the viewer to the near clipping plane
        (always positive).

    Returns
    -------
    tuple(float, float, float, float)
        Right, left, top, bottom.

    """
    scale = np.tan(angle_of_view * 0.5 * np.pi / 180) * n
    r = aspect_ratio * scale
    l = -r
    t = scale
    b = -t
    return r, l, t, b


def gl_frustum(b, t, l, r, n, f):
    """Set up the perspective projection matrix.

    Parameters
    ----------
    b : float
        Bottom.
    t : float
        Top.
    l : float
        Left.
    r : float
        Left.
    n : float
        Distance of the near plane.
    f : float
        Distance of the far plane.

    Returns
    -------
    np.ndarray
        Perspective projection matrix..

    """
    M = np.zeros((4, 4))
    M[0, 0] = 2 * n / (r - l)
    M[1, 1] = 2 * n / (t - b)

    M[2, 0] = (r + l) / (r - l)
    M[2, 1] = (t + b) / (t - b)
    M[2, 2] = -(f + n) / (f - n)
    M[2, 3] = -1

    M[3, 2] = -2 * f * n / (f - n)
    return M


def mult_points_matrix(M, P):
    """Multiply the point cloud by a transformation matrix M.

    Parameters
    ----------
    M : np.ndarray
        Transformation matrix.
    P : np.ndarray
        Point cloud.

    Returns
    -------
    np.ndarray
        Transformed point cloud.

    """
    P_ = np.matmul(M, P)
    P_[:3, :] = P_[:3, :] / P_[3, :]
    return P_


"""@tf.function(
    input_signature=(
        tf.TensorSpec([None, 4, 4], dtype=tf.float32),
        tf.TensorSpec([4, None], dtype=tf.float32)
        )
)"""
def mult_points_matrix_tf(M, P):
    """Projects points from 3D to 2D

    Parameters
    ----------
    M : tf.Tensor
        Perspective projection matrix (4 X 4) to project points from 3D to 2D.
    P : np.ndarray
        Point cloud (B X 4 X |P|).

    Returns
    -------
    tf.Tensor
        Project point cloud (B X 3 X |P|).

    """
    # M = tf.reshape(M, (None, M.shape[0], M.shape[1]))
    P_ = tf.matmul(M[:, :], P[:])
    # print(P_.shape)
    D = P_[:, 3, :]
    # print((P_[:, :3, :] / D[:, None, :]).shape)
    return P_[:, :3, :] / D[:, None, :]


"""@tf.function(
    input_signature=(
        tf.TensorSpec([4, None], dtype=tf.float32),
        tf.TensorSpec([4, 4], dtype=tf.float32),
        tf.TensorSpec([None, 4, 4], dtype=tf.float32),
        tf.TensorSpec([], dtype=tf.int32)
        )
)"""
def PtoImg2(P, M_proj, Rt, wh):
    """Perspective projection of a point cloud to multiple images. The batch
    size determines the number of images.

    Parameters
    ----------
    P : np.ndarray
        Point clouds.
    M_proj : np.ndarray
        Projection Matrices.
    Rt : np.ndarray
        Matrices to rotate and translate the points.
    wh : int
        Dimension of the output images in pixel (e.g. 256).

    Returns
    -------
    list
        Description of returned object.
    tf.Tensor
        Description of returned object.

    """
    # transform points into camera space
    # 4 X 4
    M = M_proj
    M = tf.matmul(M, Rt)
    # print(M.shape)
    # B X 3 X |P|
    P_proj = mult_points_matrix_tf(M, P)
    # print(P_proj.shape)
    # points of the transformed cloud in image space
    # B X 2 X |P|
    P_proj = P_proj[:, :2, :]
    P_proj = tf.cast(P_proj, tf.float32)
    # print(P_proj.shape)

    # idxs of the points of the transformed cloud that are in the image
    # B X |P|
    wh_f = tf.cast(wh, tf.float32)
    # print(P_proj.shape)
    P_proj = (P_proj + 1) * 0.5 * wh_f
    P_proj = tf.cast(P_proj, tf.int32)
    mask1 = tf.where(
        (P_proj[:, 0, :] > 0) &
        (P_proj[:, 0, :] < wh) &
        (P_proj[:, 1, :] > 0) &
        (P_proj[:, 1, :] < wh),
        tf.ones_like(P_proj[:, 0], tf.bool),
        tf.zeros_like(P_proj[:, 0], tf.bool)
        )

    # print(P_proj.shape)
    # print(P_proj.shape)
    # filtered = []
    n_iter = tf.shape(Rt)[0]
    #filtered = tf.TensorArray(tf.int32, size=n_iter)
    filtered = []
    #idxs = tf.TensorArray(tf.int32, size=n_iter)
    idxs = []
    for i in tf.range(n_iter):
        P_proj_x = P_proj[i, 0, :]
        x = tf.boolean_mask(P_proj_x, mask1[i])
        P_proj_y = P_proj[i, 1, :]
        y = tf.boolean_mask(P_proj_y, mask1[i])
        # v = tf.stack([x, y])
        v = tf.concat([x, y], axis=0)
        #filtered = filtered.write(i, v)
        #idxs = idxs.write(i, tf.shape(x)[0])
        filtered.append(v)
        idxs.append(tf.shape(x)[0])
    #return filtered.concat(), mask1, idxs.stack()
    return tf.concat(filtered, axis=0), mask1, tf.stack(idxs)


def grad_to_rad(angle):
    """Transforms an angle from degrees to radians.

    Parameters
    ----------
    angle : float
        Angle in degrees.

    Returns
    -------
    float
        Transforms an angle from degrees to radians.

    """
    return (angle / 180) * np.pi

def P_to_vox_grid(P, voxels_pad_or_clip_size, grid_cell=0.05):
    """Method that transforms a point cloud to a voxel grid.

    Parameters
    ----------
    P : np.ndarray
        The point cloud.
    voxels_pad_or_clip_size : int
        How many voxels should result.
    grid_cell : float
        Size of the voxels.

    Returns
    -------
    tf.Tensor, tf.Tensor, tf.Tensor
        Features for each voxel, the indices of each voxel in the grid, the number of valid voxels.

    """
    grid_cell_size = np.array([grid_cell, grid_cell, grid_cell], dtype=np.float32)
    num_valid_points = np.array([P.shape[0]], dtype=np.int32)
    grid_cell_size = tf.convert_to_tensor(grid_cell_size, dtype=tf.float32, name="grid_cell_size")
    points = tf.convert_to_tensor(P[:, :3], dtype=tf.float32, name="points")
    points = tf.expand_dims(points, axis=0)
    features = tf.convert_to_tensor(P[:, 3:], dtype=tf.float32, name="features")
    features = tf.expand_dims(features, axis=0)
    num_valid_points = tf.convert_to_tensor(num_valid_points, dtype=tf.int32, name="num_valid_points")
    voxel_features, voxel_indices, num_valid_voxels, segment_ids, voxel_start_location = pointcloud_to_sparse_voxel_grid(
            points=points, features=features, num_valid_points=num_valid_points, grid_cell_size=grid_cell_size, voxels_pad_or_clip_size=voxels_pad_or_clip_size, segment_func=tf.math.unsorted_segment_max)
    return voxel_features, voxel_indices, num_valid_voxels


def pad_or_sample_points(P, n):
    """Sample or pad points so that the point cloud P has n points.

    Parameters
    ----------
    P : np.ndarray
        The point cloud.
    n : int
        Size of the resulting point cloud.

    Returns
    -------
    np.array
        The resulting point cloud.

    """
    n_P = P.shape[0]
    if n_P > n:
        idxs = np.arange(n_P)
        idxs = np.random.choice(idxs, size=n)
        return P[idxs]
    elif n_P < n:
        n_zeros = n - n_P
        mod = n_zeros % 2
        if mod == 0:
            n_zeros = int(n_zeros / 2)
            zeros = np.zeros((n_zeros, P.shape[1]))
            return np.vstack((zeros, P, zeros))
        else:
            n_zeros = math.floor(n_zeros / 2)
            zeros = np.zeros((n_zeros, P.shape[1]))
            n_zeros_ = n_zeros + 1
            zeros_ = np.zeros((n_zeros_, P.shape[1]))
            return np.vstack((zeros, P, zeros_))
    else:
        return P


def euclidean_distance(a, b, asarray=False):
    d = np.sqrt(np.sum(np.square(a-b)))
    if asarray:
        d = np.array([d], dtype=np.float32).reshape(1,1)
    return d


def estimate_normals_curvature(P, k_neighbours=5, nn_algo="brute", verbose=False):
    n_P = P.shape[0]
    if n_P <= k_neighbours:
        k_neighbours = 5
    n = P.shape[1]

    try:
        nbrs = NearestNeighbors(n_neighbors=k_neighbours, algorithm=nn_algo, metric="euclidean").fit(P[:, :3])
        _, nns = nbrs.kneighbors(P[:, :3])
    except Exception as e:
        normals = np.zeros((n_P, 3), dtype=np.float32)
        curvature = np.zeros((n_P, ), dtype=np.float32)
        return normals, curvature, None, False


    k_nns = nns[:, 1:k_neighbours]
    p_nns = P[k_nns[:]]
    
    p = np.matlib.repmat(P, k_neighbours-1, 1)
    p = np.reshape(p, (n_P, k_neighbours-1, n))
    p = p - p_nns
    
    C = np.zeros((n_P,6))
    C[:,0] = np.sum(np.multiply(p[:,:,0], p[:,:,0]), axis=1)
    C[:,1] = np.sum(np.multiply(p[:,:,0], p[:,:,1]), axis=1)
    C[:,2] = np.sum(np.multiply(p[:,:,0], p[:,:,2]), axis=1)
    C[:,3] = np.sum(np.multiply(p[:,:,1], p[:,:,1]), axis=1)
    C[:,4] = np.sum(np.multiply(p[:,:,1], p[:,:,2]), axis=1)
    C[:,5] = np.sum(np.multiply(p[:,:,2], p[:,:,2]), axis=1)
    C /= k_neighbours
    
    normals = np.zeros((n_P,n))
    curvature = np.zeros((n_P,1))
    if verbose:
        for i in tqdm(range(n_P), desc="Compute normals"):
            C_mat = np.array([[C[i,0], C[i,1], C[i,2]],
                [C[i,1], C[i,3], C[i,4]],
                [C[i,2], C[i,4], C[i,5]]])
            values, vectors = np.linalg.eig(C_mat)
            lamda = np.min(values)
            k = np.argmin(values)
            norm = np.linalg.norm(vectors[:,k])
            if norm == 0:
                norm = 1e-12
            normals[i,:] = vectors[:,k] / np.linalg.norm(vectors[:,k])
            sum_v = np.sum(values)
            if sum_v == 0:
                sum_v = 1e-12
            curvature[i] = lamda / sum_v
    else: 
        for i in range(n_P):
            C_mat = np.array([[C[i,0], C[i,1], C[i,2]],
                [C[i,1], C[i,3], C[i,4]],
                [C[i,2], C[i,4], C[i,5]]])
            values, vectors = np.linalg.eig(C_mat)
            lamda = np.min(values)
            k = np.argmin(values)
            norm = np.linalg.norm(vectors[:,k])
            if norm == 0:
                norm = 1e-12
            normals[i,:] = vectors[:,k] / np.linalg.norm(vectors[:,k])
            sum_v = np.sum(values)
            if sum_v == 0:
                sum_v = 1e-12
            curvature[i] = lamda / sum_v

    return normals, curvature, k_nns, True