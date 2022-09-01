import datetime
import numpy as np
from abc import abstractmethod

from .utils import socket_recv, socket_send
from environment.utils import mkdir
from .base_network_master import NodeProcess, Server


class KFoldNodeProcess(NodeProcess):
    def __init__(
            self,
            conn,
            pid,
            ready_val,
            lock,
            sock,
            addr,
            buffer_size,
            recv_timeout,
            n_node_cpus,
            model_dir,
            test_interval
            ):
        super().__init__(
            conn=conn,
            pid=pid,
            ready_val=ready_val,
            lock=lock,
            sock=sock,
            addr=addr,
            buffer_size=buffer_size,
            recv_timeout=recv_timeout,
            n_node_cpus=n_node_cpus
            )
        self.model_dir = model_dir
        self.test_interval = test_interval
        self.test_msg_size = None
        self.grads_msg_size = None

    def send_update(self):
        #print("Send net")
        socket_send(file=self.model_dir + "/tmp_net.npz", sock=self.sock,
            buffer_size=self.buffer_size)
        #print("Send net done")

    def on_run_start(self):
        self.train_step = 0
        self.tested = False

    def receive_test_results(self):
        #print("recv test results")
        for i in range(self.n_node_cpus):
            if self.test_msg_size is None:
                fsize = socket_recv(file="./tmp/test_stats_" + str(self.id) + "_" + str(i) + ".npz", sock=self.sock,
                    buffer_size=self.buffer_size, timeout=self.recv_timeout)
                self.test_msg_size = fsize
            else:
                socket_recv(file="./tmp/test_stats_" + str(self.id) + "_" + str(i) + ".npz", sock=self.sock,
                    buffer_size=self.buffer_size, msg_size=self.test_msg_size)
            self.sock.send(("recv_" + str(i)).encode())
        #print("size of results file:", fsize)
        #print("stats received")

    def receive_gradients(self):
        #print("receive gradients")
        for i in range(self.n_node_cpus):
            if self.grads_msg_size is None:
                fsize = socket_recv(file="./tmp/grads_" + str(self.id) + "_" + str(i) + ".npz", sock=self.sock,
                    buffer_size=self.buffer_size, timeout=self.recv_timeout)
                self.grads_msg_size = fsize
            else:
                socket_recv(file="./tmp/grads_" + str(self.id) + "_" + str(i) + ".npz", sock=self.sock,
                    buffer_size=self.buffer_size, msg_size=self.grads_msg_size)
            self.sock.send(("recv_" + str(i)).encode())
        #print("size of gradients file:", fsize)
        #print("gradients received")

    def recv_loop(self):
        if self.train_step > 0 and self.train_step % self.test_interval == 0 and not self.tested:
            # apply a test step
            self.receive_test_results()
            self.tested = True
            return "recv_test"
        else:
            # apply a train step
            self.receive_gradients()
            self.tested = False
            self.train_step += 1
            return "recv_train"

class KFoldServer(Server):
    def __init__(
            self,
            test_interval,
            k_fold,
            model_dir,
            seed,
            p_data,
            dataset,
            ip="127.0.0.1",
            port=5000,
            buffer_size=4096,
            n_nodes=10,
            recv_timeout=4,
            experiment_name="1",
            n_epochs = 1,
            set_test_interval=None
            ):
        if k_fold < 2:
            raise Exception("Need more than k={0} folds".format(k_fold))
        mkdir("./tmp")
        self.test = False
        self.test_interval = 2
        self.k_fold = k_fold
        self.model_dir = model_dir
        self.seed = seed
        self.p_data = p_data
        self.dataset = dataset
        self.fold_stats = []
        self.experiment_name = experiment_name
        self.test_interval, self.examples_per_fold = set_test_interval(dataset=dataset, n_epochs=n_epochs)
        self.save_model()
        super().__init__(
            ip=ip,
            port=port,
            buffer_size=buffer_size,
            n_nodes=n_nodes,
            recv_timeout=recv_timeout,
            n_loops=k_fold * (self.test_interval + self.examples_per_fold)
            )

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def get_model_vars(self):
        pass

    @abstractmethod
    def preprocess_grads(self, grads):
        pass

    def on_init(self):
        pass

    def create_node_process(self, conn, pid, ready_val, lock, sock, addr, buffer_size, recv_timeout, n_node_cpus):
        return KFoldNodeProcess(
            conn=conn,
            pid=pid,
            ready_val=ready_val,
            lock=lock,
            sock=sock,
            addr=addr,
            buffer_size=buffer_size,
            recv_timeout=recv_timeout,
            n_node_cpus=n_node_cpus,
            model_dir=self.model_dir,
            test_interval=self.test_interval)

    def on_node_start(self):
        self.grads = self.n_total_cpus * [None]
        self.losses = self.n_total_cpus * [None]
        self.tresults = self.n_total_cpus * [None]

    def get_init_msg(self):
        return str(self.test_interval) + "," + str(self.k_fold) + "," + str(self.seed) + "," + self.dataset + "," + str(self.p_data)

    def store_grads(self, msg, id):
        #print("Store gradients")
        for i in range(self.cpus_per_node[id]):
            grad_data = np.load("./tmp/grads_" + str(id) + "_" + str(i) + ".npz", allow_pickle=True)
            grad_data = dict(grad_data)
            tmp_grads = []
            grad_nrs = []
            loss_dict = {}

            vars_ = self.get_model_vars()
            # check if there is no mismatch between grads and vars
            for key, value in grad_data.items():
                if key.endswith("loss"):
                    loss_dict[key] = value
                    continue
                for i in range(len(vars_)):
                    name = vars_[i].name
                    if name == key:
                        grad_nrs.append(i)
                        if vars_[i].shape != value.shape:
                            raise Exception("Shape mismatch at variable: {0}. A Shape: {1}, B Shape: {2}".format(name, vars_[i].shape, value.shape))
                tmp_grads.append(value)
            n_grads = len(tmp_grads)
            if n_grads == 0:
                raise Exception("Received no gradients")
            grads = n_grads*[None]
            #print("Store {0} gradients".format(n_grads))
            for i in range(n_grads):
                grad_nr = grad_nrs[i]
                grad = tmp_grads[i]
                grads[grad_nr] = grad
            self.grads[self.did] = grads
            self.losses[self.did] = loss_dict
            self.did += 1

    def store_test_results(self, msg, id):
        #print("master: load test results")
        for i in range(self.cpus_per_node[id]):
            test_data = np.load("./tmp/test_stats_" + str(id) + "_" + str(i) + ".npz", allow_pickle=True)
            test_data = dict(test_data)
            self.tresults[self.did] = test_data
            self.did += 1
        #print("master: done")

    def on_msg_received(self, msg, id):
        #print("master: process {0} send msg: {1} ".format(id, msg))
        if self.test:
            self.store_test_results(msg=msg, id=id)
        else:
            self.store_grads(msg=msg, id=id)

    def on_start(self):
        self.train_step = 0
        self.test_step = 0
        self.did = 0

    def on_recv(self):
        self.test = self.train_step > 0 and self.train_step % self.test_interval == 0 and not self.test

    def on_loop(self):
        # data is already received at this point
        if self.test:
            self.write_test_results()
            self.test_step += 1
            # method that resets the model 
            self.reset_method()
            self.save_model()
        else:
            self.reduce()
            self.save_model()
            self.train_step += 1
        self.did = 0

    def on_stop(self):
        avg_fs, raw_fs = self.avg_fold_stats(fold_stats=self.fold_stats)
        self.save_fold_stats(avg_fs=avg_fs, raw_fs=raw_fs)

    @abstractmethod
    def reset_method(self):
        pass

    def avg_fold_stats(self, fold_stats):
        keys = fold_stats[0][0]
        avg_fs = {}
        raw_fs = {}
        n_folds = len(fold_stats)
        for key in keys:
            avg_fs[key] = 0
            raw_fs[key] = n_folds * [0]
        for i in range(len(keys)):
            avg = 0
            key = keys[i]
            for j in range(n_folds):
                raw = fold_stats[j][1][i]
                avg += (raw / n_folds)
                raw_fs[key][j] = raw
            avg_fs[key] = avg
        return avg_fs, raw_fs

    def save_fold_stats(self, avg_fs, raw_fs):
        fname = "./" + self.dataset + "_" + self.experiment_name + ".txt"
        results = "Avg Stats\n"
        for k, v in avg_fs.items():
            results += str(k) + ": " + str(v) + "\n"
        results += "Raw Stats\n"
        for k, v in raw_fs.items():
            results += str(k) + ": "
            for i in range(len(v)-1):
                results += str(v[i]) + ", "
            results += str(v[-1]) + "\n"
        with open(fname, "w") as f:
            f.write(results)

    def avg_test_results(self):
        keys = list(self.tresults[0].keys())
        n_stats = len(keys)
        avg_stats = n_stats * [None]
        for i in range(n_stats):
            tmp_results = []
            for id in range(self.n_total_cpus):
                test_data = self.tresults[id]
                key = keys[i]
                stat = test_data[key]
                tmp_results.append(stat)
            avg_results = np.average(tmp_results, axis=0)
            avg_stats[i] = avg_results
        return keys, avg_stats

    @abstractmethod
    def write_avg_test_results(self, keys, avg_stats):
        pass

    def write_test_results(self):
        #print("write test results")
        keys, avg_stats = self.avg_test_results()
        self.fold_stats.append((keys, avg_stats))
        self.write_avg_test_results(keys=keys, avg_stats=avg_stats)
        #print("done")

    @abstractmethod
    def apply_grads(self, grads, vars_):
        pass

    def reduce(self):
        vars_ = self.get_model_vars()
        grads = self.avg_gradients()
        if len(grads) != len(vars_):
            print("")
            for i in range(len(vars_)):
                print(vars_[i].name)
            raise Exception("Number mismatch. Gradients: {0}, Vars: {1}".format(len(grads), len(vars_)))

        grads, adds = self.preprocess_grads(grads)
        
        # check if gradients match the number of variables
        for i in range(len(vars_)):
            # print(vars_[i].name, grads[i])
            if grads[i].shape != vars_[i].shape:
                raise Exception("Shape mismatch at variable {0}. A Shape: {1}, B Shape: {2}".format(vars_[i].name, grads[i].shape, vars_[i].shape))
        self.apply_grads(grads, vars_)

        self.write_train_results(adds=adds)

    @abstractmethod
    def to_tensor(self, t, dtype):
        pass

    def avg_gradients(self):
        n_grads = len(self.grads[0])
        avg_grads = n_grads*[None]
        for i in range(n_grads):
            tmp_grads = []
            for id in range(self.n_total_cpus):
                grads_id = self.grads[id]
                grad = grads_id[i]
                tmp_grads.append(grad)
            avg_grad = np.average(tmp_grads, axis=0)
            avg_grad = self.to_tensor(t=avg_grad, dtype="float32")
            avg_grads[i] = avg_grad
        avg_grads = tuple(avg_grads)
        return avg_grads

    @abstractmethod
    def write_loss(self, name, loss):
        pass

    @abstractmethod
    def write_adds(self, adds):
        pass

    @abstractmethod
    def on_train_write_end(self):
        pass

    def write_train_results(self, adds):
        keys = list(self.losses[0].keys())
        n_losses = len(keys)

        for i in range(n_losses):
            key = keys[i]
            tmp_losses = []
            for id in range(self.n_total_cpus):
                loss_dict = self.losses[id]
                loss = loss_dict[key]
                tmp_losses.append(loss)
            avg_loss = np.average(tmp_losses)
            self.write_loss(name=key, loss=avg_loss)
        self.write_adds(adds)
        self.on_train_write_end()