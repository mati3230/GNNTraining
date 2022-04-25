import numpy as np
from multiprocessing import Process
import argparse
import h5py
import os

import matplotlib.pyplot as plt

import libcp
import libply_c
import libgeo
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay

from environment.density_utils import densities_np, densities_np_osize
from environment.partition import Partition
from policies.utils import estimate_normals_curvature, get_general_bb#, pad_or_sample_points
from environment.utils import mkdir
from cpu_utils import process_range
from optimization.utils import distance_sort
from optimization.tf_utils import np_fast_dot


def plot_samples2D(P, i, center):

    m = np.ones((P.shape[0], ), dtype=np.bool)
    m[i] = False
    plt.scatter(P[m, 0], P[m, 1], label='Non Samples', color='b', s=25, marker="o")


    m = np.zeros((P.shape[0], ), dtype=np.bool)
    m[i] = True

    plt.scatter(P[m, 0], P[m, 1], label='Samples', color='r', s=25, marker="x")
    plt.scatter(center[0], center[1], label='Center', color='g', s=25, marker="*")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Test')
    plt.legend()
    plt.show()


def test(p1=60, p1_mu=0.9, p1_sigma=0.1, p2=10, p2_div = 8, p2_off=0.1, k=4, far=True, seed=10):
    """
    Visualizes nearest neighbour sampling against iterative farthest point sampling. 
    

    Parameters
    ----------
    p1 : int
        Number of points of a normal distributed point cloud p1.
    p1_mu : float
        Mean of the normal distribution p1.
    p1_sigma : float
        Standard deviation of the normal distribution p1.
    p2 : int
        Number of points of a uniform distributed point cloud p2.
    p2_div : float
        Factor to make the value range of p2 smaller.
    p2_off : float
        Translate the value range of p2.
    k : int
        Number of points that should be sampled.
    far : bool
        Sample the farthest or the nearest points. 
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    """
    create a point cloud as combination of the normal distribution p1 
    and the uniform distribution p2.
    """
    P = p1_mu + p1_sigma * np.random.randn(p1, 2)
    P_ = p2_off + np.random.rand(p2, 2) / p2_div
    P = np.vstack((P, P_))
    
    # sample the k farthest or nearest points and plot them
    center = np.mean(P, axis=0)
    i, d = get_characteristic_points(P=P, center=center, k=k, far_points=far)
    i = np.array(i, dtype=np.int32)
    plot_samples2D(P=P, i=i, center=center)

    # sample points by considering the distance from the center point
    d_sort, distances = distance_sort(P=P, p_query=center)
    if far:
        i = d_sort[-k:]
    else:
        i = d_sort[:k]
    plot_samples2D(P=P, i=i, center=center)


def get_characteristic_points(P, center, k, far_points=True):
    """
    Farthest or nearest point sampling. 
    
    Parameters
    ----------
    P : np.ndarray
        The input point cloud where points should be sampled.
    center : np.ndarray
        Center of the point cloud P.
    k : int
        number of points that should be sampled.
    far_points : bool
        If True, farthest point sampling will be applied. Nearest
        point sampling otherwise.

    Returns
    -------
    np.ndarray, np.ndarray
        Indices of the sampled points of P. The distance values
        of the sampled points.
    """
    c_sort_idx = 0
    if far_points:
        c_sort_idx = -1
    d_sort, distances = distance_sort(P=P, p_query=center)
    center_idx = int(d_sort[0])

    c_idx = center_idx
    exclude = [c_idx]
    idxs = []
    dists = []
    n_P = P.shape[0]
    if k > n_P:
        k = n_P - 1
    for i in range(k):
        d_sort, distances = distance_sort(P=P, p_query=P[c_idx])
        
        # apply mask
        mask = np.ones((d_sort.shape[0], ), dtype=np.bool)
        for j in range(len(exclude)):
            tmp_idx = np.where(d_sort == exclude[j])[0]
            mask[tmp_idx] = False
        d_sort_ = d_sort[mask]
        try:
            c_idx = int(d_sort_[c_sort_idx])
        except:
            break
        c_dist = distances[c_idx]
        idxs.append(c_idx)
        dists.append(c_dist)
        if c_idx in exclude:
            raise Exception("Sampling error")
        exclude.append(c_idx)
    return idxs, np.array(dists, dtype=np.float32)


def zero_padding(x, target_size):
    """Append zeros to a vector or an nd array (in the last dimension).

    Parameters
    ----------
    x : np.ndarray
        Input points that should be padded with zeros.
    target_size : int
        Target length of the last dimension.

    Returns
    -------
    np.ndarray
        Padded array.
    """
    if x.shape[0] < target_size:
        diff_target = target_size - x.shape[0]
        if len(x.shape) <= 1:
            return np.vstack( (x[:, None], np.zeros((diff_target, 1)) ) ).reshape(target_size, )
        else:
            target_shape = (diff_target, ) + x.shape[1:]
            #print(target_shape, target_size)
            return np.vstack( (x, np.zeros(target_shape) ) )
    else:
        return x


def get_volume(bb, off=0):
    """Calculates the volume of a bounding box in 3 dimensions.

    Parameters
    ----------
    bb : np.ndarray
        Bounding box.
    off : int
        Skip dimensions with off > 0.

    Returns
    -------
    float
        Volume of the bounding box.
    """
    volume = 1
    for i in [1,3,5]:
        volume *= (bb[i+off] - bb[i-1+off])
    return volume


def get_min_max(feature, target, pad=True):
    """Get the minimum and maximum values of a feature
    vector. 

    Parameters
    ----------
    feature : np.ndarray
        Feature vector where the minimum and maximum should be calculated. 
    target : int
        Target length of the resulting vector. One half will consist of the min
        values and the other half of the max values. 
    pad : bool
        Should zero padding be applied?

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
        An array of the minimum and the maximum features with the
        corresponding indices. 
    """
    sortation = np.argsort(feature)
    t_half = int(target / 2)
    half = t_half
    if sortation.shape[0] < target:
        if sortation.shape[0] % 2 == 0:
            half = sortation.shape[0]
        else:
            half = sortation.shape[0] - 1
        half = int(half / 2)
    
    min_idxs = sortation[:half]
    max_idxs = sortation[-half:]

    min_f = feature[min_idxs]
    max_f = feature[max_idxs]
    if pad:
        min_f = zero_padding(x=min_f, target_size=t_half)
        max_f = zero_padding(x=max_f, target_size=t_half)
    return min_f, max_f, min_idxs, max_idxs

def extract_min_max_points(P, min_idxs, max_idxs, target, center=None, center_max=None):
    """Extract the points according to indices where a
    features has minimum and maximum values.

    Parameters
    ----------
    P : np.ndarray
        The input point cloud.
    min_idxs : np.ndarray
        Indices of the minimum of a feature.
    max_idxs : np.ndarray
        Indices of the maximum of a feature.
    target : int
        Target number of points.
    center : np.ndarray
        Center of the point cloud P.
    center_max : int
        Maximum index to determine the center point. It can be used to 
        ignore translate points into the origin with center_max=3.

    Returns
    -------
    np.ndarray, np.ndarray
        Minimum and the maximum points.
    """
    min_P = P[min_idxs]
    max_P = P[max_idxs]
    if center is not None and center_max is not None:
        min_P[:, :center_max] -= center[:center_max]
        max_P[:, :center_max] -= center[:center_max]
    #print(min_P.shape, max_P.shape)
    min_P = zero_padding(x=min_P, target_size=target)
    max_P = zero_padding(x=max_P, target_size=target)
    min_P = min_P.flatten()
    max_P = max_P.flatten()
    return min_P, max_P


def hists_feature(P, center, bins=10, min_r=-0.5, max_r=0.5):
    """Create a histogram of the feature values of a point cloud.

    Parameters
    ----------
    P : np.ndarray
        The input point cloud.
    center : np.ndarray
        Center of the point cloud P.
    bins : int
        Number of bins to create the histogram.
    min_r : float
        Minimum value of the range to compute the histogram.
    max_r : float
        Maximum value of the range to compute the histogram.

    Returns
    -------
    np.ndarray
        Histogram values for every dimension. Output shape is the
        number of dimensions times bins. 
    """
    hists = np.zeros((P.shape[1] * bins))
    for i in range(P.shape[1]):
        hist, bin_edges = np.histogram(a=P[:, i]-center[i], bins=bins, range=(min_r, max_r))
        start = i * bins
        stop = start + bins
        hists[start:stop] = hist / P.shape[0]
    return hists


def compute_features_geof(cloud, geof):
    geof_mean = np.mean(geof, axis=0)
    geof_mean = 2 * (geof_mean - 0.5)

    rgb_mean = np.mean(cloud[:, 3:6], axis=0)
    
    feats = np.vstack((geof_mean[:, None], rgb_mean[:, None]))
    feats = feats.astype(np.float32)
    feats = feats.reshape(feats.shape[0], )
    return feats


def compute_features(cloud, n_curv=30, k_curv=14, k_far=30, n_normal=30, bins=10, min_r=-0.5, max_r=0.5):
    """Compute features from a point cloud P. The features of a point cloud are:
    - Mean color 
    - Median color
    - 25% Quantil of the color
    - 75% Quantil of the color
    - Standard deviation of all features
    - Average normal
    - Standard deviation of the normals
    - Median normal
    - 25% Quantil of the normals
    - 75% Quantil of the normals
    - n_normal/2 maximum angles of the normals with the x-axis
    - n_normal/2 minimum angles of the normals with the x-axis
    - n_normal/2 points that correspond to the max normal angle values
    - n_normal/2 points that correspond to the min normal angle values
    - Average curvature value
    - Standard deviation of the curvature values
    - Median curvature value
    - 25% Quantil of the curvature values
    - 75% Quantil of the curvature values
    - n_curv/2 maximum curvature values
    - n_curv/2 minimum curvature values
    - n_curv/2 points that correspond to the max curvature values
    - n_curv/2 points that correspond to the min curvature values
    - k_far farthest points
    - k_far distances of the farthest points
    - volume of the spatial bounding box
    - volume of the color bounding box
    - histograms of every dimension

    Parameters
    ----------
    P : np.ndarray
        The input point cloud.
    n_curv : int
        Number of points for the curvature feature.
    k_curv : int
        Number of neighbours in order to calculate the  curvature features.
    k_far : int
        Number of points to sample the farthest points.  
    n_normal : int
        Number of normal features.
    bins : int
        Number of bins for a histogram in every dimension of the point cloud.
    min_r : float
        Minimum value to consider to calculate the histogram.
    max_r : float
        Maximum value to consider to calculate the histogram.


    Returns
    -------
    np.ndarray
        The calculated features.
    """
    P = np.array(cloud, copy=True)
    center = np.mean(P[:, :6], axis=0)

    std = np.std(P[:, :6], axis=0)
    median_color = np.median(P[:, 3:6], axis=0)
    q_25_color = np.quantile(P[:, 3:6], 0.25, axis=0)
    q_75_color = np.quantile(P[:, 3:6], 0.75, axis=0)
    
    #normals, curv, _, ok = estimate_normals_curvature(P=P[:, :3], k_neighbours=k_curv)
    normals = P[:, 7:]
    curv = P[:, 6]
    P[:, :3] -= center[:3]
    #max_P = np.max(np.abs(P[:, :3]))
    #P[:, :3] /= (max_P + 1e-6)
    ########normals########
    # mean, std
    mean_normal = np.mean(normals, axis=0)
    std_normal = np.std(normals, axis=0)
    median_normal = np.median(normals, axis=0)
    q_25_normal = np.quantile(normals, 0.25, axis=0)
    q_75_normal = np.quantile(normals, 0.75, axis=0)
    #print("mean: {0}\nstd: {1}\nmedian: {2}\nq25: {3}\nq75: {4}".format(mean_normal.shape, std_normal.shape, median_normal.shape, q_25_normal.shape, q_75_normal.shape))

    # min, max
    ref_normal = np.zeros((normals.shape[0], 3))
    ref_normal[:, 0] = 1
    normal_dot, _, _ = np_fast_dot(a=ref_normal, b=normals)
    min_normal, max_normal, min_n_idxs, max_n_idxs = get_min_max(feature=normal_dot, target=n_normal)
    min_P_n, max_P_n = extract_min_max_points(P=P[:, :6], min_idxs=min_n_idxs, max_idxs=max_n_idxs, target=int(n_normal/2), center=center, center_max=3)
    #print("min f: {0}\nmax f: {1}".format(min_normal.shape, max_normal.shape))
    #print("min P: {0}\nmax P: {1}".format(min_P_n.shape, max_P_n.shape))

    ########curvature########
    # mean, std
    curv = curv.reshape(curv.shape[0], )
    mean_curv = 2 * (np.mean(curv) - 0.5)
    mean_curv = np.array([mean_curv], dtype=np.float32)
    std_curv = np.std(curv)
    std_curv = np.array([std_curv], dtype=np.float32)
    median_curv = np.median(curv)
    median_curv = 2 * (median_curv - 0.5)
    median_curv = np.array([median_curv], dtype=np.float32)
    q_25_curv = np.quantile(curv, 0.25)
    q_25_curv = 2 * (q_25_curv - 0.5)
    q_25_curv = np.array([q_25_curv], dtype=np.float32)
    q_75_curv = np.quantile(curv, 0.75)
    q_75_curv = 2 * (q_75_curv - 0.5)
    q_75_curv = np.array([q_75_curv], dtype=np.float32)
    #print("mean: {0}\nstd: {1}\nmedian: {2}\nq25: {3}\nq75: {4}".format(mean_curv.shape, std_curv.shape, median_curv.shape, q_25_curv.shape, q_75_curv.shape))
    
    # min, max
    min_curv, max_curv, min_c_idxs, max_c_idxs = get_min_max(feature=curv, target=n_curv, pad=False)
    min_curv = 2 * (min_curv - 0.5)
    max_curv = 2 * (max_curv - 0.5) 
    target_c = int(n_curv / 2)
    min_curv = zero_padding(x=min_curv, target_size=target_c)
    max_curv = zero_padding(x=max_curv, target_size=target_c)
    min_P_c, max_P_c = extract_min_max_points(P=P[:, :6], min_idxs=min_c_idxs, max_idxs=max_c_idxs, target=target_c, center=center, center_max=3)
    #print("min f: {0}\nmax f: {1}".format(min_curv.shape, max_curv.shape))
    #print("min P: {0}\nmax P: {1}".format(min_P_c.shape, max_P_c.shape))

    ########farthest points########
    idxs, dists = get_characteristic_points(P=P[:, :6], center=center, k=k_far, far_points=True)
    far_points = P[idxs, :6]
    #far_points[:, :3] -= center[:3]
    far_points = zero_padding(x=far_points, target_size=k_far)
    far_points = far_points.flatten()
    dists = zero_padding(x=dists, target_size=k_far)
    far_normals = normals[idxs]
    far_normals = zero_padding(x=far_normals, target_size=k_far)
    far_normals = far_normals.flatten()
    far_curv = curv[idxs]
    far_curv = zero_padding(x=far_curv, target_size=k_far)


    ########volumes########
    bb = get_general_bb(P=P[:, :6])
    spatial_volume = get_volume(bb=bb, off=0)
    color_volume = get_volume(bb=bb, off=6)
    volumes = np.array([spatial_volume, color_volume])
    volumes -= 0.5
    volumes *= 2


    ########hists########
    hists = hists_feature(P=P[:, :6], center=center, bins=bins, min_r=min_r, max_r=max_r)
    hists -= 0.5
    hists *= 2
    #print(hists)

    ########concatenation########
    features = np.vstack(
        (
        center[3:6, None],#3,abs
        median_color[:, None],#3,abs
        q_25_color[:, None],#3,abs
        q_75_color[:, None],#3,abs
        std[:, None],#6
        
        mean_normal[:, None],#3
        std_normal[:, None],#3
        median_normal[:, None],#3,abs
        q_25_normal[:, None],
        q_75_normal[:, None],
        
        min_normal[:, None],#n_normal*3
        max_normal[:, None],#n_normal*3
        
        min_P_n[:, None],#n_normal*6,local
        max_P_n[:, None],#n_normal*6,local
        
        mean_curv[:, None],#1
        std_curv[:, None],#1
        median_curv[:, None],#3,abs
        q_25_curv[:, None],
        q_75_curv[:, None],
        min_curv[:, None],#n_curv/2
        max_curv[:, None],#n_curv/2
        min_P_c[:, None],#6*n_curv/2,local
        max_P_c[:,None],#6*n_curv/2,local
        
        far_points[:, None],#k_far*6,local
        far_curv[:, None],
        far_normals[:, None],
        dists[:, None],#k_far
        volumes[:, None],#2
        hists[:, None]#6*bins
        ))
    features = features.astype(dtype=np.float32)
    features = features.reshape(features.shape[0], )
    return features


def feature_point_cloud(P, k_curv=14):
    xyz = np.ascontiguousarray(P[:, :3], dtype=np.float32)
    _, target_fea = compute_graph_nn_2(xyz, k_curv, k_curv)
    feats = libply_c.compute_curv_normals(xyz, target_fea, k_curv, False)
    curv = feats[:, 5]
    normals = feats[:, 5:]
    center = np.mean(P[:, :3], axis=0)
    P[:, :3] -= center
    max_P = np.max(np.abs(P[:, :3]))
    P[:, :3] /= (max_P + 1e-6)
    P[:, 3:] /= 255
    P[:, 3:] -= 0.5
    P[:, 3:] *= 2
    P = np.hstack((P, curv[:, None], normals))
    return P, center


def compute_graph_nn_2(xyz, k_nn1, k_nn2, voronoi = 0.0):
    #print("Compute Graph NN")
    """compute simulteneoulsy 2 knn structures
    only saves target for knn2
    assumption : knn1 <= knn2"""
    assert k_nn1 <= k_nn2, "knn1 must be smaller than knn2"
    n_ver = xyz.shape[0]
    #compute nearest neighbors
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn2+1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    del nn
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    #---knn2---
    target2 = (neighbors.flatten()).astype('uint32')
    #---knn1-----
    if voronoi>0:
        tri = Delaunay(xyz)
        graph["source"] = np.hstack((tri.vertices[:,0],tri.vertices[:,0], \
              tri.vertices[:,0], tri.vertices[:,1], tri.vertices[:,1], tri.vertices[:,2])).astype('uint64')
        graph["target"]= np.hstack((tri.vertices[:,1],tri.vertices[:,2], \
              tri.vertices[:,3], tri.vertices[:,2], tri.vertices[:,3], tri.vertices[:,3])).astype('uint64')
        graph["distances"] = ((xyz[graph["source"],:] - xyz[graph["target"],:])**2).sum(1)
        keep_edges = graph["distances"]<voronoi
        graph["source"] = graph["source"][keep_edges]
        graph["target"] = graph["target"][keep_edges]
        
        graph["source"] = np.hstack((graph["source"], np.matlib.repmat(range(0, n_ver)
            , k_nn1, 1).flatten(order='F').astype('uint32')))
        neighbors = neighbors[:, :k_nn1]
        graph["target"] =  np.hstack((graph["target"],np.transpose(neighbors.flatten(order='C')).astype('uint32')))
        
        edg_id = graph["source"] + n_ver * graph["target"]
        
        dump, unique_edges = np.unique(edg_id, return_index = True)
        graph["source"] = graph["source"][unique_edges]
        graph["target"] = graph["target"][unique_edges]
       
        graph["distances"] = graph["distances"][keep_edges]
    else:
        neighbors = neighbors[:, :k_nn1]
        distances = distances[:, :k_nn1]
        graph["source"] = np.matlib.repmat(range(0, n_ver)
            , k_nn1, 1).flatten(order='F').astype('uint32')
        graph["target"] = np.transpose(neighbors.flatten(order='C')).astype('uint32')
        graph["distances"] = distances.flatten().astype('float32')
    #save the graph
    #print("Done")
    return graph, target2


def superpoint_graph(xyz, rgb, k_nn_adj=10, k_nn_geof=45, lambda_edge_weight=1, reg_strength=0.1, d_se_max=0, n_repair=4):
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    #---compute 10 nn graph-------
    graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)
    #---compute geometric features-------
    #print("Compute geof")
    geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof, False).astype('float32')
    del target_fea

    senders = graph_nn["source"]
    receivers = graph_nn["target"]
    distances = graph_nn["distances"]

    uni_verts, direct_neigh_idxs, n_edges = np.unique(senders, return_index=True, return_counts=True)
    senders = senders.astype(np.uint32)
    receivers = receivers.astype(np.uint32)
    uni_verts = uni_verts.astype(np.uint32)
    direct_neigh_idxs = direct_neigh_idxs.astype(np.uint32)
    n_edges = n_edges.astype(np.uint32)

    mask = libgeo.unidirectional(uni_verts, direct_neigh_idxs, n_edges, receivers)
    mask = mask.astype(np.bool)
    #print(mask.shape, mask.dtype)
    senders = senders [mask]
    receivers = receivers[mask]
    distances = distances[mask]

    graph_nn["source"] = senders
    graph_nn["target"] = receivers
    graph_nn["distances"] = distances

    #choose here which features to use for the partition
    #print(np.max(rgb))
    features = np.hstack((geof, rgb/255.)).astype("float32")#add rgb as a feature for partitioning
    features[:,3] = 2. * features[:,3] #increase importance of verticality (heuristic)

    verbosity_level = 0.0
    speed = 2.0
    store_bin_labels = 0
    cutoff = 0 
    spatial = 0 
    weight_decay = 1
    graph_nn["edge_weight"] = np.array(1. / ( lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = "float32")

    components, in_component, stats, duration = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"], 
        graph_nn["edge_weight"], reg_strength, cutoff, spatial, weight_decay, verbosity_level, speed, store_bin_labels)
    #print("Done")
    #print(components)
    #print(in_component)
    components = np.array(components, dtype = "object")
    n_com = max(in_component)+1

    in_component = np.array(in_component)

    sp_centers = np.zeros((n_sps, 3), dtype=np.float32)
    for i in range(n_com):
        idxs = components[i]
        sp_centers[i] = np.mean(P[idxs, :3], axis=0)

    assigned_partition_vec = np.zeros((xyz.shape[0], ), dtype=np.uint32)
    for i in range(n_com):
        idxs = components[i]
        assigned_partition_vec[idxs] = i

    source = graph_nn["source"].astype(np.uint32)
    target = graph_nn["target"].astype(np.uint32)
    senders, receivers = libgeo.components_graph(assigned_partition_vec, source, target)

    # make each edge bidirectional
    tmp_senders = np.array(senders, copy=True)
    senders = np.hstack((senders[None, :], receivers[None, :]))
    receivers = np.hstack((receivers[None, :], tmp_senders[None, :]))

    # unique: filter multiple occurance of the same edges
    edges = np.vstack((senders, receivers))
    uni_edges = np.unique(edges, axis=1)
    senders = uni_edges[0, :]
    receivers = uni_edges[1, :]

    # sorted them according to the senders 
    sortation = np.argsort(senders)
    senders = senders[sortation]
    receivers = receivers[sortation]

    uni_senders, senders_idxs, senders_counts = np.unique(senders, return_index=True, return_counts=True)

    ####################################
    if n_com != uni_senders.shape[0]:
        # identify which nodes do not have senders (i.e. single nodes)
        single_nodes = []
        for i in range(n_com):
            if i in uni_senders:
                continue
            single_nodes.append(i)
        print("Have to connect {0} nodes".format(len(single_nodes)))
        single_nodes = np.array(single_nodes, dtype=np.uint32)
        # determine the nearest neighbours of the single nodes
        nn = NearestNeighbors(n_neighbors=n_repair, algorithm="kd_tree").fit(sp_centers)
        single_dists, single_neigh = nn.kneighbors(single_nodes)

        offset = n_repair - 1
        n_add = single_nodes.shape[0] * offset
        n_senders = np.zeros((n_add, ), dtype=np.uint32)
        n_receivers = np.zeros((n_add, ), dtype=np.uint32)
        for i in range(single_nodes.shape[0]):
            single_node = single_nodes[i]
            start = i * offset
            stop = start + offset
            n_senders[start:stop] = single_node
            n_receivers[start:stop] = single_neigh[i, 1:]

        # make new edges bidirectional
        tmp_n_senders = np.array(n_senders, copy=True)
        n_senders = np.hstack((n_senders[None, :], n_receivers[None, :]))
        n_receivers = np.hstack((n_receivers[None, :], tmp_n_senders[None, :]))

        # add new edges to the old ones
        senders = np.vstack((senders[:, None], n_senders[:, None]))
        receivers = np.vstack((receivers[:, None], n_receivers[:, None]))

        senders = senders.reshape((senders.shape[0], ))
        receivers = receivers.reshape((senders.shape[0], ))

        # sorted them according to the senders 
        sortation = np.argsort(senders)
        senders = senders[sortation]
        receivers = receivers[sortation]

        uni_senders, senders_idxs, senders_counts = np.unique(senders, return_index=True, return_counts=True)

    # double check
    if n_com != uni_senders.shape[0]:
        raise Exception("Different number of superpoints and unique senders ({0}, {1})".format(n_com, uni_senders.shape[0]))
    ######################################
    #print("{0} edges filtered, {1} unique edges".format(n_filtered, len(uni_edges)))
    return n_com, senders.shape[0], components, senders, receivers, uni_senders, senders_idxs, senders_counts, geof, sp_centers, sp_centers


def process_scenes(id, args, min_i, max_i):
    """Iterate over the superpoints of the scenes and calculate
    feature vectors for each superpoint. The features will be
    stored in a h5py file for every scene.

    Parameters
    ----------
    id : int
        The process id.
    args : dict
        Additional arguments that are needed to calculate the features.
    min_i : int
        Minimum scene index.
    max_i : float
        Maximum scene index.
    """
    #print(id, max_i, min_i, max_i - min_i)
    scenes = args["scenes"]
    dataset = args["dataset"]
    n_ft = None
    sp_sizes = []
    for i in range(min_i, max_i):
        # e.g. ./S3DIS_Scenes/Area1_conferenceRoom_1/P.npz
        scene = scenes[i]
        strs = scene.split("/")
        area_room_name = strs[-2]

        data = np.load(scene)

        P = data["P"]
        #print("point cloud has {0} points".format(P.shape[0]))
        
        partition_vec = data["partition_vec"]
        partition_uni = data["partition_uni"]
        partition_idxs = data["partition_idxs"]
        partition_counts = data["partition_counts"]

        p_gt = Partition(partition=partition_vec, uni=partition_uni, idxs=partition_idxs, counts=partition_counts)
        
        # n_com, n_sedg, components, senders, receivers, uni_senders, senders_idxs, senders_counts
        n_sps, n_edges, sp_idxs, senders, receivers, uni_senders, senders_idxs, senders_counts, geof, sp_centers = superpoint_graph(
            xyz=P[:, :3],
            rgb=P[:, 3:],
            reg_strength=0.3)
        #print("created {0} superpoints, have {1} uni senders, have {2} senders".format(n_sps, uni_senders.shape[0], senders.shape[0]))

        assigned_partition_vec = np.zeros((P.shape[0], ), np.int32)
        for j in range(n_sps):
            idxs = np.unique(sp_idxs[j])
            sp_idxs[j] = idxs
            #hf.create_dataset("sp_idx_"+str(j), data=idxs, dtype=np.uint32)
            assigned_partition_vec[idxs] = j

        p_a = Partition(partition=assigned_partition_vec)
        densities = p_gt.compute_densities(p_a, densities_np)
        alpha = p_a.alpha(densities)

        #print("Superpoint graph has {0} nodes and {1} edges".format(n_sps, n_edges))
        #print("Compute features for every superpoint")
        P, center = feature_point_cloud(P=P)
        """
        hf.create_dataset("P", data=P)
        hf.create_dataset("partition_vec", data=partition_vec)
        hf.create_dataset("assigned_partition_vec", data=assigned_partition_vec, dtype=np.int32)
        hf.create_dataset("n_sps", data=np.array([n_sps], dtype=np.int32))
        """
        if n_ft is None:
            sp_idxs_ = sp_idxs[0]
            sp = P[sp_idxs_]
            #features = compute_features(cloud=sp)
            features = compute_features_geof(cloud=sp, geof=geof)
            n_ft = features.shape[0]
            print("feature vector has size of {0}".format(n_ft))
        all_features = np.zeros((n_sps, n_ft), dtype=np.float32)
        sps_sizes = []
        for k in range(n_sps):
            sp_idxs_ = sp_idxs[k]
            #hf.create_dataset(str(k), data=sp_idxs_)
            sp = P[sp_idxs_]
            sps_sizes.append(sp_idxs_.shape[0])
            #features = compute_features(cloud=sp)
            features = compute_features_geof(cloud=sp, geof=geof)
            all_features[k] = features
        mean_sps = np.mean(sps_sizes)
        sp_sizes.append(mean_sps)
        std_sps = np.std(sps_sizes)
        """
        hf.create_dataset("mean_sps", data=np.array([mean_sps], dtype=np.float32))
        hf.create_dataset("std_sps", data=np.array([std_sps], dtype=np.float32))
        hf.create_dataset("n_edges", data=np.array([n_edges], dtype=np.int32))
        """
        unions = []
        uni_edges = []
        n_filtered = 0
        unions = np.zeros((senders.shape[0], ), dtype=np.bool)
        for k in range(senders.shape[0]):
            S_i = senders[k]
            S_j = receivers[k]
            union = alpha[S_i] == alpha[S_j]
            #print(S_i, S_j, union)
            unions[k] = union

        distances = np.sqrt(np.sum((sp_centers[senders] - sp_centers[receivers])**2, axis=1))

        ######################
        n_unions = unions.shape[0]
        false_edge_idxs = np.where(unions == False)[0]
        n_false_edges = false_edge_idxs.shape[0]
        n_true_edges = n_unions - n_false_edges
        true_edge_idxs = np.delete(np.arange(n_unions), false_edge_idxs)
        if n_false_edges <= n_true_edges:
            n_samples = n_false_edges * 2
            true_edge_idxs = true_edge_idxs[:n_false_edges]
        else:
            n_samples = n_true_edges * 2
            false_edge_idxs = false_edge_idxs[:n_true_edges]
        if false_edge_idxs.shape[0] != true_edge_idxs.shape[0]:
            raise Exception("True and false idxs have different shapes ({0}, {1})".format(false_edge_idxs.shape[0], true_edge_idxs[0]))
        batch_size = 16
        if n_samples <= batch_size:
            raise Exception("do not have enough samples, got: {0}, want: {1}".format(n_samples, batch_size))
        n_batches = math.floor(n_samples / batch_size)
        #sample_size = batch_size / 2
        sample_idxs = np.arange(false_edge_idxs.shape[0], dtype=np.uint32)
        np.random.shuffle(sample_idxs)
        b_half = batch_size / 2
        for j in range(n_batches):
            start = j * b_half
            stop = start + b_half
            s_idxs = sample_idxs[start:stop]
            s_false_edge_idxs = false_edge_idxs[s_idxs]
            s_true_edge_idxs = true_edge_idxs[s_idxs]
            edge_idxs = np.vstack(s_false_edge_idxs[:, None], s_true_edge_idxs[:, None])
            edge_idxs = edge_idxs.reshape(edge_idxs.shape[0], )

            sampled_unions = unions[edge_idxs]
        
            sampled_senders = senders[edge_idxs]
            sampled_receivers = receivers[edge_idxs]
            sampled_distances = distances[edge_idxs]

            #uni_senders = np.unique(sampled_senders)
            #uni_receivers = np.unique(sampled_receivers)

            sp_idxs = np.vstack((sampled_senders[:, None], sampled_receivers[:, None]))
            sp_idxs = sp_idxs.reshape(sp_idxs.shape[0], )
            sp_idxs = np.unique(sp_idxs)

            receivers = receivers.astype(np.uint32)
            senders_idxs = senders_idxs.astype(np.uint32)
            senders_counts = senders_counts.astype(np.uint32)
            distances = distances.astype(np.float32)
            sp_idxs = sp_idxs.astype(np.uint32)
            depth = 3
            n_verts = uni_senders.shape[0]
            senders_, receivers_, distances_ = libgeo.geodesic_neighbours(sp_idxs, senders_idxs, senders_counts,
                receivers, distances, depth, n_verts, False)
            all_inter_idxs = np.zeros((batch_size, ), dtype=np.int32)
            # we search the original edges in the subgraph
            for k in range(batch_size):
                source = sampled_senders[k]
                target = sampled_receivers[k]
                s_idxs = np.where(senders_ == source)[0]
                r_idxs = np.where(receivers_ == target)[0]
                inter_idxs = np.intersect1d(s_idxs, r_idxs)
                if inter_idxs.shape[0] != 1:
                    """
                    print(senders)
                    print(receivers)
                    print(senders_)
                    print(receivers_)
                    """
                    print(sampled_senders, sampled_receivers)
                    print(sp_idxs)
                    print(source, target)
                    print("extracted {0} edges, soure idxs: {1}, recv idxs {2}".format(senders_.shape[0], s_idxs.shape[0], r_idxs.shape[0]))
                    raise Exception("Faulty intersection: shape 0 of idxs should be 1 got {0}".format(inter_idxs.shape))
                all_inter_idxs[k] = inter_idxs[0].astype(np.int32)
            #"""
            #print(np.min(all_inter_idxs), np.max(all_inter_idxs), all_inter_idxs.shape, senders_.shape)
            y = y[edge_idxs]

            all_nodes = np.vstack((senders_[:, None], receivers_[:, None]))
            all_nodes = all_nodes.reshape(all_nodes.shape[0], )
            all_nodes = np.unique(all_nodes)

            tmp_node_features = np.zeros((all_nodes.shape[0], all_features.shape[-1]))
            mapping = 0
            mapped_senders = np.array(senders_, copy=True)
            mapped_receivers = np.array(receivers_, copy=True)
            for i in range(all_nodes.shape[0]):
                node_idx = all_nodes[i]
                mapped_senders[senders_ == node_idx] = mapping
                mapped_receivers[receivers_ == node_idx] = mapping
                tmp_node_features[i] = all_features[node_idx]
                mapping += 1
            node_features = tmp_node_features
            node_features = node_features.astype(np.float32)
            mapped_senders = mapped_senders.astype(np.uint32)
            mapped_receivers = mapped_receivers.astype(np.uint32)

            # e.g. ./s3dis/graphs/Area1_conferenceRoom_1.h5 will be stored as new file
            hf = h5py.File("{0}/graphs/{1}_{2}.h5".format(dataset, area_room_name, j), "w")
            hf.create_dataset("node_features", data=node_features)
            hf.create_dataset("senders", data=mapped_senders)
            hf.create_dataset("receivers", data=mapped_receivers)
            hf.create_dataset("unions", data=sampled_unions)
            hf.create_dataset("edge_idxs", data=all_inter_idxs)
            hf.close()
            #return {"nodes": node_features, "senders": mapped_senders,
            #    "receivers": mapped_receivers, "edges": None, "globals": None}, y, all_inter_idxs
        ######################

        #print(n_sps)
        """
        hf.create_dataset("node_features", data=all_features)
        hf.create_dataset("senders", data=senders)
        hf.create_dataset("receivers", data=receivers)
        hf.create_dataset("unions", data=unions)
        hf.create_dataset("uni_senders", data=uni_senders)
        hf.create_dataset("senders_idxs", data=senders_idxs)
        hf.create_dataset("senders_counts", data=senders_counts)
        hf.create_dataset("sp_centers", data=sp_centers)
        hf.create_dataset("distances", data=distances)
        """

        #print("unions: {0}".format(np.mean(unions)))

        progress = 100*(i-min_i+1)/(max_i - min_i)
        print("{0}\t{1:.2f}\t{2}\t{3}".format(id, progress, area_room_name, n_ft))
        #break
        #return
    print("id: {0}, mean and std superpoint size of {1} scenes: {2:.2f}, {3:.2f}".format(id, len(scenes), np.mean(sp_sizes), np.std(sp_sizes)))


def main(args):
    wargs = {}
    scenes = []
    for area_room in os.listdir(args.dataset):
        # e.g. ./S3DIS_Scenes/Area1_conferenceRoom_1/P.npz
        scenes.append(args.dataset + "/" + area_room + "/P.npz")
    #scenes = scenes[:10]
    wargs["scenes"] = scenes
    wargs["dataset"] = args.out_dataset

    mkdir(args.out_dataset + "/graphs")
    # print("PID\tProgress\tScene\t|S|")
    process_range(workload=len(scenes), n_cpus=args.n_cpus, process_class=Process, target=process_scenes, args=wargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="./S3DIS_Scenes")
    parser.add_argument(
        "--out_dataset",
        type=str,
        default="./s3dis")
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=1,
        help="n cpus used for the cache creation")
    args = parser.parse_args()
    main(args=args)