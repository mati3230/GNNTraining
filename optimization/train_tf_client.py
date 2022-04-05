import numpy as np
import tensorflow as tf
import os
import math
from multiprocessing import Value
from abc import abstractmethod

from .train_test_network_client import TrainTestWorker, TrainTestClient
from .base_trainer import BaseTrainer
from .utils import get_type
from .tf_utils import setup_gpu


class TFWorker(TrainTestWorker):
    def __init__(
            self,
            conn,
            id,
            ready_val,
            lock,
            test_interval,
            train_p,
            seed,
            dataset,
            dataset_dir,
            args_file,
            p_data,
            train_idxs,
            test_idxs,
            start_with_work=True,
            verbose=False):
        self.verbose = verbose
        tf.random.set_seed(seed)
        super().__init__(
            conn=conn,
            id=id,
            ready_val=ready_val,
            lock=lock,
            test_interval=test_interval,
            train_p=train_p,
            seed=seed,
            dataset=dataset,
            dataset_dir=dataset_dir,
            args_file=args_file,
            p_data=p_data,
            train_idxs=train_idxs,
            test_idxs=test_idxs,
            start_with_work=start_with_work,
            )

    def on_init(self):
        setup_gpu()

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

        # we use the BaseTrainer to conveniently load the json args file
        trainer = BaseTrainer(self.args_file, "types.json", compose_env_args=compose_env_args)

        self.batch_size = trainer.params["batch_size"]

        np.random.seed(self.seed)

        # calculate the nr of batches
        self.n_batches = math.floor(self.train_idxs.shape[0] / self.batch_size)
        self.n_t_batches = self.get_n_t_batches()
        self.batch_id = 0

        # init the neural net
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
        policy_type = get_type(trainer.params["policy_path"], trainer.params["policy_type"])
        self.model = policy_type(**policy_args)
        self.data_files, self.n_files = self.load_dataset(dir=self.dataset_dir, p_data=self.p_data)
        batch = self.load_batch(i=0, train_idxs=self.train_idxs, dir=self.dataset_dir,
            files=self.data_files, batch_size=self.batch_size)
        self.prediction(batch=batch)
        self.model.reset()
        print("neural net ready")

        # create arrays for the batches that are used in training and testing

        self.on_init_end()

    def get_n_t_batches(self):
        return math.floor(self.test_idxs.shape[0] / self.batch_size)

    def load_model(self, dir, name):
        # abstract method to load the variables
        self.model.load(directory=dir, filename=name)

    def train(self):
        # training phase
        #print("Compute gradients")
        with tf.GradientTape() as tape:
            batch = self.load_batch(i=self.batch_id, train_idxs=self.train_idxs, dir=self.dataset_dir,
                files=self.data_files, batch_size=self.batch_size)
            
            losses = self.compute_losses(batch=batch)
            if self.verbose:
                print(losses)
            loss = losses["loss"]

            vars_ = tape.watched_variables()
            grads = tape.gradient(loss, vars_)

            # save gradients
            save_dict = {}
            for i in range(len(grads)):
                grad = grads[i]
                grad_np = grad.numpy()
                name = vars_[i].name
                save_dict[name] = grad_np
            for k, v in losses.items():
                save_dict[k] = v.numpy()
            np.savez("./tmp/grads_" + str(self.id) + ".npz", **save_dict)

        self.batch_id += 1
        if self.batch_id >= self.n_batches:
            self.batch_id = 0

    def test_prediction(self, i):
        batch = self.load_batch(i=i, train_idxs=self.test_idxs, dir=self.dataset_dir, 
            files=self.data_files, batch_size=self.batch_size)
        action = self.prediction(batch=batch)
        action = action.numpy()
        action = np.squeeze(action)
        y = batch[1]
        return y, action

    def test(self):
        accs = []
        precs = []
        recs = []
        f1s = []
        #print("Compute test")
        for i in range(self.n_t_batches):
            y, action = self.test_prediction(i=i)
            TP = np.sum((y == 1) & (action == 1))
            TN = np.sum((y == 0) & (action == 0))
            FN = np.sum((y == 1) & (action == 0))
            FP = np.sum((y == 0) & (action == 1))
            acc = (TP + TN) / y.shape[0]
            if TP + FP == 0:
                prec = TP / (TP + FP + 1e-12)
            else:
                prec = TP / (TP + FP)
            if TP + FN == 0:
                rec = TP / (TP + FN + 1e-12)
            else:
                rec = TP / (TP + FN)
            if prec + rec == 0:
                f1 = 2*(prec * rec)/(prec + rec + 1e-12)
            else:
                f1 = 2*(prec * rec)/(prec + rec)
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
        save_dict = {}
        save_dict["Mean_Acc"] = np.mean(accs)
        save_dict["Mean_Prec"] = np.mean(precs)
        save_dict["Mean_Rec"] = np.mean(recs)
        save_dict["Mean_F1"] = np.mean(f1s)
        save_dict["Std_Acc"] = np.std(accs)
        save_dict["Std_Prec"] = np.std(precs)
        save_dict["Std_Rec"] = np.std(recs)
        save_dict["Std_F1"] = np.std(f1s)
        if self.verbose:
            print(save_dict)
        np.savez("./tmp/test_stats_" + str(self.id) + ".npz", **save_dict)

    def load_dataset(self, dir, p_data):
        data_files = os.listdir(dir)
        len_data_files = len(data_files)
        if p_data < 1:
            n_data = math.floor(p_data * len_data_files)
            data_files = data_files[:n_data]
            len_data_files = len(data_files)
        return data_files, len_data_files

    @abstractmethod
    def prediction(self, batch):
        pass

    @abstractmethod
    def compute_losses(self, batch):
        pass

    @abstractmethod
    def on_init_end(self):
        pass

    @abstractmethod
    def load_example(self, dir, files, idx, data_scale):
        pass

    @abstractmethod
    def load_batch(self, i, train_idxs, dir, files, batch_size, data_scale):
        pass


class TFClient(TrainTestClient):
    def __init__(
            self,
            server_ip,
            server_port,
            n_cpus,
            shared_value,
            critical_mem,
            init_buffer=4096,
            data_dir="./tmp",
            dataset_dir="./s3dis/graphs",
            buffer_size=4096):
        super().__init__(
            server_ip=server_ip,
            server_port=server_port,
            n_cpus=n_cpus,
            shared_value=shared_value,
            critical_mem=critical_mem,
            init_buffer=init_buffer,
            data_dir=data_dir,
            dataset_dir=dataset_dir,
            buffer_size=buffer_size)

    def unpack_msg(self, msg, i):
        super().unpack_msg(msg=msg, i=i) 
        self.args_file = msg[i+6]
        print("args_file: {0}".format(self.args_file))

    def load_dataset(self, dir, p_data):
        data_files = os.listdir(dir)
        len_data_files = len(data_files)
        if p_data < 1:
            n_data = math.floor(p_data * len_data_files)
            data_files = data_files[:n_data]
            len_data_files = len(data_files)
        return data_files, len_data_files

    def get_data(self):
        return self.load_dataset(dir=self.files_dir, p_data=self.p_data)

    @abstractmethod
    def get_worker(self, conn, id, ready_val, lock, train_idxs, test_idxs):
        pass