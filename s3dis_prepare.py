#from s3dis_provider import DataProvider
import argparse
import os
import numpy as np
from tqdm import tqdm
from environment.utils import mkdir, file_exists, render_point_cloud


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


def prepare_scenes(dataset_name):
    """Method to prepare the scenes.

    Parameters
    ----------
    dataset_name : str
        Name of the s3dis data set to create a blacklist.
    """
    mkdir("./S3DIS_Scenes")
    s3dis_dir = os.environ["S3DIS_DIR"] + "/data"
    special_objects = get_special_objects()

    for dir in os.listdir(s3dis_dir):
        area_dir = s3dis_dir + "/" + dir
        if not os.path.isdir(area_dir):
            continue
        area_scenes = os.listdir(area_dir)
        for ai in tqdm(range(len(area_scenes)), desc="Process Area"):
            scene = area_scenes[ai]
            scene_name = "Area" + dir[-1] + "_" + scene
            # print(scene_name)
            n_scene_dir = "./S3DIS_Scenes/" + scene_name
            if file_exists(n_scene_dir + "/P.npz"):
                continue
            scene_dir = area_dir + "/" + scene + "/Annotations"
            if not os.path.isdir(scene_dir):
                continue
            O = len(special_objects) + 1
            P = np.zeros((0, 6), np.float32)
            partition_vec = np.zeros((0, 1), np.int32)
            mkdir(n_scene_dir)
            ok = True
            for obj_file in os.listdir(scene_dir):
                if len(obj_file) <= 4:
                    continue
                if not obj_file.endswith(".txt"):
                    continue
                obj_dir = scene_dir + "/" + obj_file
                try:
                    P_O = np.loadtxt(obj_dir, delimiter=" ")
                except:
                    print("Error with file {0}".format(obj_dir))
                    raise
                try:
                    P = np.vstack((P, P_O))
                except Exception as e:
                    print("Error in scene", n_scene_dir, ": ", e)
                    blacklist = open(dataset_name + "_blacklist.txt", "a")
                    blacklist.write("\n")
                    blacklist.write(n_scene_dir)
                    blacklist.close()
                    ok = False
                    break
                p_vec = np.ones((P_O.shape[0], 1), np.int32)
                label = obj_file.split("_")[0]
                if label in special_objects:
                    p_vec *= special_objects[label]
                else:
                    p_vec *= O
                    O += 1
                partition_vec = np.vstack((partition_vec, p_vec))
            if ok:
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

                np.savez(n_scene_dir + "/P.npz", P=P, partition_vec=partition_vec, partition_uni=partition_uni, partition_idxs=partition_idxs, partition_counts=partition_counts)


def main():
    """Program entry point. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="precalc",
        help="options: precalc, visualize_all, visualize_single")
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
    print(args)
    print("mode:", args.mode)
    if args.mode == "visualize_single":
        # TODO: load scenes via scene directories
        dat_p = DataProvider(verbose=args.verbose)
        dat_p.id = args.scene
        print("scene:", args.scene)
        P, partition_vec, id = dat_p.get_cloud_and_partition()
        print(id, P.shape, partition_vec.shape)
        # n_objects = np.unique(partition_vec).shape[0]
        render_point_cloud(P=P, animate=args.animate)
        render_point_cloud(
            P=P, partition_vec=partition_vec, animate=args.animate)
        if args.render_segs:
            render_all_segments(P=P, partition_vec=partition_vec, animate=args.animate)
         # """
    else:
        prepare_scenes("s3dis")
        if args.mode != "visualize_all":
            return
        dat_p = DataProvider(verbose=args.verbose)
        dat_p.select_id()
        #nP_mean = []
        for i in range(len(dat_p.scenes)):
            P, partition_vec, id, _, _, _ = dat_p.get_cloud_and_partition()
            #if P.shape[0] < 1000000:
            #    nP_mean.append(P.shape[0])
            #else:
            #    dat_p.add_id_to_blacklist()
            #"""
            print(id, P.shape, partition_vec.shape, "progress:", i, "/", len(dat_p.scenes))
            if args.mode == "visualize_all":
                if args.use_scene:
                    if dat_p.id == args.scene:
                        args.use_scene = False
                    else:
                        dat_p.select_id()
                        continue
                # render_point_cloud(P=P, animate=args.animate, width=960)
                render_point_cloud(P=P, animate=args.animate, width=960, left=960)
                render_point_cloud(
                    P=P, partition_vec=partition_vec, animate=args.animate, width=960, left=960)
                # render_point_cloud(P=P[:, :3], partition_vec=partition_vec, animate=args.animate)
                #"""
            dat_p.select_id()
        #print(np.mean(nP_mean), "(", np.std(nP_mean), ")")
        #print(len(nP_mean), "/", len(dat_p.scenes))


if __name__ == "__main__":
    main()
