import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
import argparse
from graph_nets import utils_tf
from multiprocessing import Value
import os
import math
import h5py

from optimization.train_tf_client import TFWorker, TFClient
from optimization.kfold_tf_client import KFoldTFWorker, KFoldTFClient
from optimization.utils import load_graph_batch
from optimization.tf_utils import contrastive_loss, euclidean_distance


class KFoldTFWorkerGraph(KFoldTFWorker):
    def __init__(
            self,
            conn,
            id,
            ready_val,
            lock,
            test_interval,
            k_fold,
            seed,
            dataset,
            dataset_dir,
            args_file,
            p_data,
            train_idxs,
            test_idxs,
            start_with_work=True,
            verbose=False):
        super().__init__(
            conn=conn,
            id=id,
            ready_val=ready_val,
            lock=lock,
            test_interval=test_interval,
            k_fold=k_fold,
            seed=seed,
            dataset=dataset,
            dataset_dir=dataset_dir,
            args_file=args_file,
            p_data=p_data,
            train_idxs=train_idxs,
            test_idxs=test_idxs,
            start_with_work=start_with_work,
            verbose=verbose
            )
        self.batch_size = 1

    def prediction(self, batch):
        graphs = batch[0]
        edge_idxs = batch[2]
        #print("prediction", edge_idxs.shape)
        #edge_idxs = tf.cast(edge_idxs, tf.int32)
        #print(edge_idxs)
        pi_action = self.model.action(obs=graphs, training=False, edge_idxs=edge_idxs)
        action = pi_action["action"]
        return action

    def compute_losses(self, batch):
        graphs, y, edge_idxs = batch
        #print("train")
        pi_action = self.model.action(obs=graphs, training=True, edge_idxs=edge_idxs)
        probs = pi_action["probs"]
        #print("done")
        probs_loss = tf.reduce_mean(probs)
        #bce_loss = self.bce(y, probs)

        y_ = tf.cast(y, tf.float32)
        probs = tf.cast(probs, tf.float32)
        #print(probs.shape)
        #f_neg = 1
        f_neg = 1.5
        pos_bce_loss = -y_ * tf.math.log(tf.where(probs == 0, probs+1e-6, probs)) * (2 - f_neg)
        neg_inp = 1-probs
        neg_bce_loss =  -(1-y_) * tf.math.log(tf.where(neg_inp == 0, neg_inp+1e-6, neg_inp)) * f_neg 
        bce_loss = pos_bce_loss + neg_bce_loss
        bce_loss = tf.reduce_mean(bce_loss)
        pos_bce_loss = tf.reduce_mean(pos_bce_loss)
        neg_bce_loss = tf.reduce_mean(neg_bce_loss)
        #tf.print(probs, summarize=-1)
        #print(pos_bce_loss, neg_bce_loss)

        #y__ = 2 * (y_ - 0.5)

        #mse = tf.reduce_mean(tf.square(y__ - probs))
        
        #"""
        #dot = pi_action["dot"]        
        #hinge_loss = 1 - (y__ * dot)
        #hinge_loss = tf.reduce_mean(hinge_loss)

        #dot_loss = (1 / (tf.math.square(dot) + 1e-6))-1
        #dot_loss = tf.reduce_mean(dot_loss)

        sum_ = 0
        vars_ = self.model.get_vars()
        for var in vars_:
            sum_ += tf.reduce_sum(tf.square(var))
        weights_loss = 0 * 0.5 * sum_
        weights_loss = tf.cast(weights_loss, tf.float32)
        bce_loss = tf.cast(bce_loss, tf.float32)

        loss = bce_loss + weights_loss
        #loss = mse + weights_loss

        #action = pi_action["action"]
        #action_loss = tf.reduce_mean(tf.cast(action, tf.float32))

        return {
            "loss": loss,
            "bce_loss": bce_loss,
            #"norm_loss": norm_loss,
            "weights_loss": weights_loss,
            #"action_loss": action_loss,
            #"hinge_loss": hinge_loss,
            #"contrastive_loss": c_loss,
            "probs_loss": probs_loss,
            "pos_bce_loss": pos_bce_loss,
            "neg_bce_loss": neg_bce_loss#,
            #"dot_loss": dot_loss,
            #"mse_loss": mse
            }

    def on_init_end(self):
        #self.bce = BinaryCrossentropy(from_logits=False)
        return

    def get_n_t_batches(self):
        return self.test_idxs.shape[0]

    def load_example(self, dir, files, idx):
        #print("load_example with {0} links".format(self.n_links))
        return load_graph_batch(i=idx, dir=dir, files=files)

    def load_batch(self, i, train_idxs, dir, files, batch_size):
        return load_graph_batch(i=i, dir=dir, files=files)

    def load_folds(self, k_fold_dir, train_folds, test_fold):
        test_fold_fname = k_fold_dir + "/" + str(test_fold) + ".h5"
        train_folds_fnames = [k_fold_dir + "/" + str(train_fold) + ".h5" for train_fold in train_folds]

        hf_test = h5py.File(test_fold_fname, "w")
        test_areas = hf_test["area_names"].copy()
        hf_test.close()

        train_areas = []
        for fname in train_folds_fnames:
            hf_train = h5py.File(fname, "w")
            train_area = hf_test["area_names"].copy()
            hf_train.close()
            train_areas.extend(train_area)
        subgraph_dir = self.dataset_dir + "../subgraphs"
        use_subgraph = os.path.exists(subgraph_dir)
        if use_subgraph:
            files = os.listdir(subgraph_dir)
            train_files = []
            for train_area in train_areas:
                for i in range(len(files)):
                    file = files[i]
                    if file.starts_with(train_area):
                        train_files.append(subgraph_dir + "/" + file)
        else:
            train_files = [self.dataset_dir + "../graphs/" + area_name + "_0.h5" for area_name in train_files]
        test_files = [self.dataset_dir + "../graphs/" + area_name + "_0.h5" for area_name in test_areas]

        return train_files, test_files


class KFoldTFClientGraph(KFoldTFClient):
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
            buffer_size=4096,
            verbose=False):
        self.verbose = verbose
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

    def load_folds(self, k_fold_dir, train_folds, test_fold):
        test_fold_fname = k_fold_dir + "/" + str(test_fold) + ".h5"
        train_folds_fnames = [k_fold_dir + "/" + str(train_fold) + ".h5" for train_fold in train_folds]

        hf_test = h5py.File(test_fold_fname, "w")
        test_areas = hf_test["area_names"].copy()
        hf_test.close()

        train_areas = []
        for fname in train_folds_fnames:
            hf_train = h5py.File(fname, "w")
            train_area = hf_test["area_names"].copy()
            hf_train.close()
            train_areas.extend(train_area)
        subgraph_dir = self.dataset_dir + "../subgraphs"
        use_subgraph = os.path.exists(subgraph_dir)
        if use_subgraph:
            files = os.listdir(subgraph_dir)
            train_files = []
            for train_area in train_areas:
                for i in range(len(files)):
                    file = files[i]
                    if file.starts_with(train_area):
                        train_files.append(subgraph_dir + "/" + file)
        else:
            train_files = [self.dataset_dir + "../graphs/" + area_name + "_0.h5" for area_name in train_files]
        test_files = [self.dataset_dir + "../graphs/" + area_name + "_0.h5" for area_name in test_areas]

    def get_worker(self, conn, id, ready_val, lock, train_idxs, test_idxs):
        return KFoldTFWorkerGraph(
            conn=conn,
            id=id,
            ready_val=ready_val,
            lock=lock,
            test_interval=self.test_interval,
            k_fold=self.k_fold,
            seed=self.seed,
            dataset=self.dataset,
            dataset_dir=self.dataset_dir,
            args_file=self.args_file,
            p_data=self.p_data,
            train_idxs=train_idxs,
            test_idxs=test_idxs,
            verbose=self.verbose)


class TFWorkerGraph(TFWorker):
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
            verbose=verbose
            )
        self.batch_size = 1

    def prediction(self, batch):
        graphs = batch[0]
        edge_idxs = batch[2]
        #print("prediction", edge_idxs.shape)
        #edge_idxs = tf.cast(edge_idxs, tf.int32)
        #print(edge_idxs)
        pi_action = self.model.action(obs=graphs, training=False, edge_idxs=edge_idxs)
        action = pi_action["action"]
        return action

    def compute_losses(self, batch):
        graphs, y, edge_idxs = batch
        #print("train")
        pi_action = self.model.action(obs=graphs, training=True, edge_idxs=edge_idxs)
        probs = pi_action["probs"]
        #print("done")
        probs_loss = tf.reduce_mean(probs)
        #bce_loss = self.bce(y, probs)

        y_ = tf.cast(y, tf.float32)
        probs = tf.cast(probs, tf.float32)
        #print(probs.shape)
        #f_neg = 1
        f_neg = 1.5
        pos_bce_loss = -y_ * tf.math.log(tf.where(probs == 0, probs+1e-6, probs)) * (2 - f_neg)
        neg_inp = 1-probs
        neg_bce_loss =  -(1-y_) * tf.math.log(tf.where(neg_inp == 0, neg_inp+1e-6, neg_inp)) * f_neg 
        bce_loss = pos_bce_loss + neg_bce_loss
        bce_loss = tf.reduce_mean(bce_loss)
        pos_bce_loss = tf.reduce_mean(pos_bce_loss)
        neg_bce_loss = tf.reduce_mean(neg_bce_loss)
        #tf.print(probs, summarize=-1)
        #print(pos_bce_loss, neg_bce_loss)

        #y__ = 2 * (y_ - 0.5)

        #mse = tf.reduce_mean(tf.square(y__ - probs))
        
        #"""
        #dot = pi_action["dot"]        
        #hinge_loss = 1 - (y__ * dot)
        #hinge_loss = tf.reduce_mean(hinge_loss)

        #dot_loss = (1 / (tf.math.square(dot) + 1e-6))-1
        #dot_loss = tf.reduce_mean(dot_loss)

        sum_ = 0
        vars_ = self.model.get_vars()
        for var in vars_:
            sum_ += tf.reduce_sum(tf.square(var))
        weights_loss = 0 * 0.5 * sum_
        weights_loss = tf.cast(weights_loss, tf.float32)
        bce_loss = tf.cast(bce_loss, tf.float32)

        loss = bce_loss + weights_loss
        #loss = mse + weights_loss

        #action = pi_action["action"]
        #action_loss = tf.reduce_mean(tf.cast(action, tf.float32))

        return {
            "loss": loss,
            "bce_loss": bce_loss,
            #"norm_loss": norm_loss,
            "weights_loss": weights_loss,
            #"action_loss": action_loss,
            #"hinge_loss": hinge_loss,
            #"contrastive_loss": c_loss,
            "probs_loss": probs_loss,
            "pos_bce_loss": pos_bce_loss,
            "neg_bce_loss": neg_bce_loss#,
            #"dot_loss": dot_loss,
            #"mse_loss": mse
            }

    def on_init_end(self):
        #self.bce = BinaryCrossentropy(from_logits=False)
        return

    def get_n_t_batches(self):
        return self.test_idxs.shape[0]

    def load_example(self, dir, files, idx):
        #print("load_example with {0} links".format(self.n_links))
        return load_graph_batch(i=idx, dir=dir, files=files)

    def load_batch(self, i, train_idxs, dir, files, batch_size):
        return load_graph_batch(i=i, dir=dir, files=files)


class TFClientGraph(TFClient):
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
            buffer_size=4096,
            verbose=False):
        self.verbose = verbose
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

    def get_worker(self, conn, id, ready_val, lock, train_idxs, test_idxs):
        return TFWorkerGraph(
            conn=conn,
            id=id,
            ready_val=ready_val,
            lock=lock,
            test_interval=self.test_interval,
            train_p=self.train_p,
            seed=self.seed,
            dataset=self.dataset,
            dataset_dir=self.dataset_dir,
            args_file=self.args_file,
            p_data=self.p_data,
            train_idxs=train_idxs,
            test_idxs=test_idxs,
            verbose=self.verbose)


def main():
    print("you have to set the ip, port, n_clients, client_id")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",type=str,default="192.168.0.164",help="IP of the server")
    parser.add_argument("--port",type=int,default=5000,help="Port of the server")
    parser.add_argument("--buffer_size",type=int,default=4096,help="Size of the transmission data")
    parser.add_argument("--n_cpus",type=int,default=1,help="Nr of cpus that will be used for the training")
    parser.add_argument("--dataset_dir",type=str,default="./s3dis/graphs",help="path to the dataset")
    parser.add_argument("--critical_mem",type=int,default=85,help="Threshold - training will stop if too much memory is used") 
    parser.add_argument("--gpu",type=bool,default=False,help="Should gpu be used")
    parser.add_argument("--verbose",type=bool,default=False,help="Print training progress")
    parser.add_argument("--k_fold",type=bool,default=False,help="Use k fold cross validation")
    args = parser.parse_args()
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.k_fold:
        KFoldTFClientGraph(
            server_ip=args.ip,
            server_port=args.port,
            n_cpus=args.n_cpus,
            dataset_dir=args.dataset_dir,
            buffer_size=args.buffer_size,
            shared_value=Value("i", True),
            critical_mem=args.critical_mem,
            verbose=args.verbose)
    else:
        TFClientGraph(
            server_ip=args.ip,
            server_port=args.port,
            n_cpus=args.n_cpus,
            dataset_dir=args.dataset_dir,
            buffer_size=args.buffer_size,
            shared_value=Value("i", True),
            critical_mem=args.critical_mem,
            verbose=args.verbose)


if __name__ == "__main__":
    main()