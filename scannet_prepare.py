import open3d as o3d
import json
import os
import numpy as np
from environment.utils import render_point_cloud, mkdir, file_exists, render_all_segments
from multiprocessing import Process
import argparse
import math


def load_json(file):
    """Read a json file.

    Parameters
    ----------
    file : str
        Path to a json file.

    Returns
    -------
    dict
        Contents of the json file.
    """
    with open(file) as f:
        dict = json.load(f)
    return dict


def load_seg_groups(filepath, filename):
    """Read the segmentation groups.

    Parameters
    ----------
    filepath : str
        Path to a json file.
    filename : str
        Name of the json file.

    Returns
    -------
    dict
        Segmentation groups.
    """
    file = filepath + filename
    dict = load_json(file)
    seg_groups = dict["segGroups"]
    return seg_groups


def load_seg_indices(filepath, filename):
    """Read the segmentation indices.

    Parameters
    ----------
    filepath : str
        Path to a json file.
    filename : str
        Name of the json file.

    Returns
    -------
    dict
        Segmentation indices.
    """
    file = filepath + filename
    dict = load_json(file)
    seg_indices = dict["segIndices"]
    seg_indices = np.array(seg_indices, dtype=np.int32)
    return seg_indices


def load_mesh(filepath, filename):
    """Load a triangle mesh file from disk.

    Parameters
    ----------
    filepath : str
        Path to a ply file.
    filename : str
        Name of the ply file.

    Returns
    -------
    o3d.geometry.Mesh
        Mesh.
    """
    file = filepath + filename
    mesh = o3d.io.read_triangle_mesh(file)
    return mesh


def get_object_mesh(seg_group, seg_indices, mesh):
    """Load a triangle mesh file from disk.

    Parameters
    ----------
    seg_group : list(int)
        Superpoint values.
    seg_indices : list(int)
        Indices of the superpoints.
    mesh : o3d.geometry.Mesh
        Mesh of a ground truth object.

    Returns
    -------
    o3d.geometry.Mesh
        Mesh.
    """
    superpoints = seg_group["segments"]
    superpoints = np.array(superpoints, dtype=np.int32)
    idxs = np.zeros((0, 1), dtype=np.int32)
    for i in range(superpoints.shape[0]):
        superpoint = superpoints[i]
        idxs_i = np.where(seg_indices == superpoint)[0]
        idxs_i = idxs_i.reshape(idxs_i.shape[0], 1)
        idxs = np.vstack((idxs, idxs_i))
    idxs = idxs.reshape(idxs.shape[0], )
    O_mesh = mesh.select_by_index(indices=idxs)
    return O_mesh


def merge_vectors(a, b):
    """Concatenate two vectors.

    Parameters
    ----------
    a : o3d.utility.Vector3dVector
        For instance, an array of points.
    b : o3d.utility.Vector3dVector
        For instance, an array of points.

    Returns
    -------
    o3d.utility.Vector3dVector
        Concatenated vector.
    """
    np_a = np.asarray(a)
    np_b = np.asarray(b)
    np_c = np.concatenate((np_a, np_b))
    return o3d.utility.Vector3dVector(np_c)


def merge_clouds(pcd_a, pcd_b):
    """Concatenate two point clouds.

    Parameters
    ----------
    pcd_a : o3d.geometry.PointCloud
        A point cloud.
    pcd_b : o3d.geometry.PointCloud
        A point cloud.

    Returns
    -------
    o3d.geometry.PointCloud
        Concatenated point clouds.
    """
    points = merge_vectors(pcd_a.points, pcd_b.points)
    colors = merge_vectors(pcd_a.colors, pcd_b.colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = points
    pcd.colors = colors
    return pcd


def sample_scene(pid, scenes, scannet_dir, upsampling, mscannet_dir, special_objects):
    """Transforms scenes from the scannet data set into point clouds with a corresponding
    ground truth partition. 

    Parameters
    ----------
    pid : int
        A process id.
    scenes : list(str)
        A list of scenes.
    scannet_dir : str
        Directory of the original scannet scenes.
    upsampling : int
        Upsampling factor n. A resulting point cloud will have n times #vertices of a mesh.
    mscannet_dir : str
        Directory where the point clouds will be stored.
    special_objects : dict
        Dictionary where some objects have special numbers such as the floor or ceiling.
    """
    for k in range(len(scenes)):
        scene = scenes[k]
        scene_dir = mscannet_dir + "/" + scene
        print("pid {0}, scene {1}/{2}".format(pid, k+1, len(scenes)))
        if file_exists(scene_dir + "/P.npz"):
            continue
        O = len(special_objects) + 1
        filepath = scannet_dir + "/" + scene + "/"
        seg_indices = load_seg_indices(filepath, scene + "_vh_clean_2.0.010000.segs.json")
        mesh = load_mesh(filepath, scene + "_vh_clean_2.ply")
        seg_groups = load_seg_groups(filepath, scene + ".aggregation.json")
        pcd = o3d.geometry.PointCloud()
        n = len(seg_groups)
        partition_vec = np.zeros((0, 1), dtype=np.int32)
        for i in range(len(seg_groups)):
            seg_group = seg_groups[i]
            label = seg_group["label"]
            O_mesh = get_object_mesh(seg_group, seg_indices, mesh)
            # o3d.visualization.draw_geometries([O_mesh], window_name=label)
            number_of_points = upsampling*len(O_mesh.vertices)
            O_pcd = O_mesh.sample_points_poisson_disk(number_of_points=number_of_points)
            # o3d.visualization.draw_geometries([pcd], window_name=label)
            pcd = merge_clouds(pcd, O_pcd)
            # print("progress: {:.2f}%".format(100*(i+1)/n))
            tmp_partition_vec = np.ones((number_of_points, 1), dtype=np.int32)
            if label in special_objects:
                tmp_partition_vec *= special_objects[label]
            else:
                tmp_partition_vec *= O
                O += 1
            partition_vec = np.vstack((partition_vec, tmp_partition_vec))
        P = np.asarray(pcd.points)
        C = 255*np.asarray(pcd.colors)
        P = np.hstack((P, C))
        xyz_mean = np.mean(P[:, :3], axis=0)
        P[:, :3] = P[:, :3] - xyz_mean

        partition_vec = partition_vec.reshape(partition_vec.shape[0], )
        #print(partition_vec.shape)
        sortation = np.argsort(partition_vec)
        #print(sortation.shape)
        P = P[sortation, :]
        #print(P.shape)
        partition_vec = partition_vec[sortation]
        #print(partition_vec.shape)
        partition_uni, partition_idxs, partition_counts = np.unique(partition_vec, return_index=True, return_counts=True)
        #print(partition_uni)

        mkdir(scene_dir)
        # TODO store as h5py (xyz, rgb, objects, ...)
        np.savez(scene_dir + "/P.npz", P=P, partition_vec=partition_vec, partition_uni=partition_uni, partition_idxs=partition_idxs, partition_counts=partition_counts)


def pick_points(pcd, width=1920):
    """Opens a open3D visualization where two points can be selected. """
    print("")
    print(
        "1) Please pick at least two correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=width)
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def get_special_objects():
    """Create a dictionary where explicit object names assigned to reseverd numbers. 

    Returns
    -------
    boolean
        True if the file exists.
    """
    special_objects = {}
    sobjs = ["wall", "floor", "ceiling"]
    for i in range(len(sobjs)):
        obj = sobjs[i]
        special_objects[obj] = i + 1
    special_objects["floor"] = special_objects["wall"]
    return special_objects


def main():
    """Program entry point. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="precalc",
        help="options: precalc, visualize_all, visualize_single")
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=10,
        help="more cpus speed up the precalc process")
    parser.add_argument(
        "--upsampling",
        type=int,
        default=4,
        help="an object will have upsampling times #vertices points")
    parser.add_argument(
        "--scene",
        type=str,
        default="Area1_office_30",
        help="scene from the scannet dataset")
    parser.add_argument(
        "--use_scene",
        type=bool,
        default=False,
        help="used if all scenes are visualized")
    parser.add_argument(
        "--render_segs",
        type=bool,
        default=False,
        help="flag to render every superpoint of scene")
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="flag to print output")
    parser.add_argument(
        "--animate",
        type=bool,
        default=False,
        help="If True, the point cloud will be animated")
    args = parser.parse_args()

    scannet_dir = os.environ["SCANNET_DIR"] + "/scans"
    mscannet_dir = "./Scannet_Scenes"
    mkdir(mscannet_dir)
    scenes = os.listdir(scannet_dir)
    special_objects = get_special_objects()

    if args.mode == "precalc":
        if args.n_cpus > 1:
            len_scenes = len(scenes)
            processes = []
            intervals = math.floor(len_scenes / args.n_cpus)
            for i in range(args.n_cpus):
                min_i = i*intervals
                max_i = min_i + intervals
                if max_i > len_scenes:
                    diff = max_i - len_scenes
                    max_i -= diff
                if max_i == min_i:
                    continue
                #scenes, scannet_dir, upsampling, mscannet_dir
                p_scenes = scenes[min_i:max_i]
                p = Process(target=sample_scene, args=(i, p_scenes, scannet_dir, args.upsampling, mscannet_dir, special_objects))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            return
        sample_scene(0, scenes, scannet_dir, args.upsampling, mscannet_dir, special_objects)
        return
    if args.mode == "visualize_all":
        for k in range(len(scenes)):
            scene = scenes[k]
            scene_dir = mscannet_dir + "/" + scene
            if not file_exists(scene_dir + "/P.npz"):
                continue
            print("pid {0}, scene {1}/{2}".format(0, k+1, len(scenes)))
            data = np.load(scene_dir + "/P.npz", allow_pickle=True)
            P = data["P"]
            partition_vec = data["partition_vec"]
            render_point_cloud(P=P, animate=args.animate)
            partition_pcd = render_point_cloud(
                P=P, partition_vec=partition_vec, animate=args.animate)
            if args.render_segs:
                render_all_segments(P=P, partition_vec=partition_vec, animate=args.animate)
            """point_idxs = pick_points(partition_pcd, width=1920)
            if len(point_idxs) == 2:
                point1 = P[point_idxs[0], :3]
                point2 = P[point_idxs[1], :3]
                dist = np.linalg.norm(point1 - point2)
                print(dist)"""

    if args.mode == "visualize_single":
        scene_dir = mscannet_dir + "/" + args.scene
        if not file_exists(scene_dir + "/P.npz"):
            return
        data = np.load(scene_dir + "/P.npz", allow_pickle=True)
        P = data["P"]
        #max_v = abs(np.max(P[:,:3]))
        #P[:,:3] = P[:,:3] / max_v
        #P[:,:3] += 0.5
        partition_vec = data["partition_vec"]
        render_point_cloud(P=P, animate=args.animate)
        partition_pcd = render_point_cloud(
            P=P, partition_vec=partition_vec, animate=args.animate)
        if args.render_segs:
            render_all_segments(P=P, partition_vec=partition_vec, animate=args.animate)


if __name__ == "__main__":
    main()
