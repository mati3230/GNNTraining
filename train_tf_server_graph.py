import tensorflow as tf
import argparse
import os
from graph_nets import utils_tf

from optimization.train_tf_server import TFServer
from optimization.kfold_tf_server import KFoldTFServer
from optimization.utils import load_graph_batch
from optimization.base_trainer import BaseTrainer
from optimization.utils import get_type, save_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",type=str,default="192.168.0.164",help="IP of the server")
    parser.add_argument("--port",type=int,default=5000,help="Port of the server")
    parser.add_argument("--buffer_size",type=int,default=4096,help="Size of the transmission data")
    parser.add_argument("--n_clients",type=int,default=1,help="Number of clients")
    parser.add_argument("--args_file",type=str,default="args_files/super_args_ffgn.json",help="file (relative path) to default configuration")
    parser.add_argument("--recv_timeout", type=int, default=8, help="Timeout to receive data stream.")
    parser.add_argument("--dataset",type=str,default="s3dis",help="options: scannet, s3dis")
    parser.add_argument("--p_data",type=float,default=1,help="Percentage of the data that should be used")
    parser.add_argument("--gpu",type=bool,default=False,help="Should gpu be used")
    parser.add_argument("--k_fold",type=bool,default=False,help="Use k fold cross validation")
    parser.add_argument("--n_epochs",type=int,default=14,help="Number of epochs for k fold cross validation")
    parser.add_argument("--experiment_name",type=str,default="1",help="Name of the experiment for output file of k fold cross validation")
    args = parser.parse_args()
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def compose_env_args(params):
            if "data_provider_path" in params:
                if params["data_provider_path"] != "":
                    #data_prov_type = get_type(
                    #    params["data_provider_path"], "DataProvider")
                    data_prov_type = None
                    data_prov_args = {
                        "max_scenes": params["max_scenes"],
                        "train_mode": params["train_mode"],
                        "train_p": params["train_p"],
                        "n_cpus": params["n_cpus"],
                        "max_P_size": params["max_P_size"],
                        "transform": params["transform"],
                        "filter_special_objects": params["filter_special_objects"]
                    }
                    scene_args = {}
                    env_args = {
                        "data_prov_type": data_prov_type,
                        "data_prov_args": data_prov_args,
                        "scene_args": scene_args,
                        "object_size": params["object_size"],
                        "delta": params["delta"],
                        "error_thres": params["error_thres"]
                    }
                    return env_args
                else:
                    raise ValueError("data_provider_path is empty.")
            else:
                return {}

    trainer = BaseTrainer(args.args_file, "types.json", compose_env_args=compose_env_args)

    obs_size = trainer.params["observation_size"]
    n_actions = trainer.params["n_actions"]

    check_numerics = trainer.params["check_numerics"]
    policy_args = {
        "name": "target_policy",
        "n_ft_outpt": trainer.params["n_ft_outpt"],
        "n_actions": n_actions,
        "observation_size": obs_size,
        "seed": trainer.params["seed"],
        "initializer": trainer.params["initializer"],
        "mode": "half",
        "check_numerics": check_numerics,
        "discrete": True}
    seed = trainer.params["seed"]
    tf.random.set_seed(seed)
    policy_type = get_type(trainer.params["policy_path"], trainer.params["policy_type"])
    print("instantiate neural net")
    policy = policy_type(**policy_args)
    print("load example")
    files_dir = "./" + args.dataset + "/graphs"
    files = os.listdir(files_dir)

    input_graphs, y, _ = load_graph_batch(i=0, dir=files_dir, files=files)
    print("prepare example with {0} elements".format(y.shape[0]))
    #return
    #input_graphs = utils_tf.data_dicts_to_graphs_tuple([dd])
    print("prediction")
    policy.action(obs=input_graphs, training=False)
    print("reset")
    policy.reset()
    print("neural net ready")
    #print(policy.get_vars()[0])
    #return

    model_dir = "./models/" + trainer.params["env_name"] + "/" + trainer.params["model_name"]

    global_norm_t = trainer.params["global_norm"]
    learning_rate = trainer.params["learning_rate"]
    test_interval = trainer.params["test_interval"]
    train_p = trainer.params["train_p"]

    if args.k_fold:
        folds_dir = "./" + args.dataset + "/folds"
        k_fold = len(os.listdir(folds_dir))
        print("k fold with {0} folds".format(k_fold))
        policy.save(directory=model_dir, filename="init_net")
        def set_test_interval(dataset, n_epochs):
            # TODO change for subgraph
            fold_dir = "./" + dataset + "/folds"
            n_folds = len(os.listdir(fold_dir))
            fold_file = fold_dir + "/0.h5"
            hf = h5py.File(fold_file, "r")
            examples_per_fold = len(list(hf["area_names"]))
            hf.close()
            test_interval = examples_per_fold * (n_folds - 1) * n_epochs
            print("set test interval to {0}".format(test_interval))
            return test_interval, examples_per_fold
        tf_server = KFoldTFServer(
            args_file=args.args_file,
            model=policy,
            global_norm_t=global_norm_t,
            learning_rate=learning_rate,
            test_interval=test_interval,
            k_fold=k_fold,
            model_dir=model_dir,
            seed=seed,
            p_data=args.p_data,
            dataset=args.dataset,
            ip=args.ip,
            port=args.port,
            buffer_size=args.buffer_size,
            n_nodes=args.n_clients,
            recv_timeout=args.recv_timeout,
            experiment_name=args.experiment_name,
            n_epochs=args.n_epochs,
            set_test_interval=set_test_interval)
    else:
        tf_server = TFServer(
            args_file=args.args_file,
            model=policy,
            global_norm_t=global_norm_t,
            learning_rate=learning_rate,
            test_interval=test_interval,
            train_p=train_p,
            model_dir=model_dir,
            seed=seed,
            p_data=args.p_data,
            dataset=args.dataset,
            ip=args.ip,
            port=args.port,
            buffer_size=args.buffer_size,
            n_nodes=args.n_clients,
            recv_timeout=args.recv_timeout)
    print("tf server initialized")
    save_config(tf_server.log_dir, str(trainer.params))
    tf_server.start()

if __name__ == "__main__":
    main()