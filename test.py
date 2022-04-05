import numpy as np
import numpy as np
import open3d as o3d
import time
import libply_c

from policies.utils import estimate_normals_curvature
from create_feature_ds import compute_graph_nn_2, compute_features, feature_point_cloud


def main():
    mesh = o3d.io.read_triangle_mesh("D:/Projects/PartitionAlgorithms/code/sn017600.ply")
    mesh.compute_adjacency_list()
    xyz = np.asarray(mesh.vertices)
    rgb = np.asarray(mesh.vertex_colors)
    P = np.hstack((xyz, rgb))
    knn = 30

    P, _ = feature_point_cloud(P)
    sp_idx = np.arange(100, dtype=np.uint32)
    sp = P[sp_idx]
    feats = compute_features(sp)
    print(feats.shape[0])

    """
    print("Compute normals")
    t1 = time.time()
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    _, target_fea = compute_graph_nn_2(xyz, knn, knn)
    feats = libply_c.compute_curv_normals(xyz, target_fea, knn, False)
    curv = feats[:, 5]
    normals = feats[:, 5:]
    print(normals.shape)
    print(curv.shape)
    t2 = time.time()
    print("Duration {0:.3f} seconds".format(t2 - t1))
    return
    print("Compute normals")
    t1 = time.time()
    normals, curv, _, ok = estimate_normals_curvature(P=P[:, :3], k_neighbours=30)
    t2 = time.time()
    print("Duration {0:.3f} seconds".format(t2 - t1))
    """



if __name__ == "__main__":
    main()