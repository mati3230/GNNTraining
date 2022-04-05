import tensorflow as tf
import socket
import argparse
from multiprocessing import Process, Pipe, Lock, Value
import time
import numpy as np
import datetime

from environment.utils import mkdir
from .utils import socket_send, socket_recv


class ClientProcess(Process):
    def __init__(self, conn, pid, ready_val, lock, sock, addr, buffer_size, model_dir, recv_timeout, test_interval):
        super().__init__()
        self.conn = conn
        self.id = pid
        self.ready_val = ready_val
        self.lock = lock
        self.sock = sock
        self.addr = addr
        self.buffer_size = buffer_size
        self.model_dir = model_dir
        self.recv_timeout = recv_timeout
        self.test_interval = test_interval
        print("Init:", self.id)
        self.work = True
        self.test_msg_size = None
        self.grads_msg_size = None

    def release_lock(self):
        time.sleep(0.2)
        self.lock.release()

    def send_update(self):
        #print("Send net")
        socket_send(file=self.model_dir + "/tmp_net.npz", sock=self.sock,
            buffer_size=self.buffer_size)
        #print("Send net done")

    def receive_test_results(self):
        #print("Recv test results")
        if self.test_msg_size is None:
            fsize = socket_recv(file="./tmp/test_stats_" + str(self.id) + ".npz", sock=self.sock,
                buffer_size=self.buffer_size, timeout=self.recv_timeout)
            self.test_msg_size = fsize
        else:
            socket_recv(file="./tmp/test_stats_" + str(self.id) + ".npz", sock=self.sock,
                buffer_size=self.buffer_size, msg_size=self.test_msg_size)
        #print("size of results file:", fsize)
        #print("Stats received")

    def receive_gradients(self):
        #print("Receive gradients")
        if self.grads_msg_size is None:
            fsize = socket_recv(file="./tmp/grads_" + str(self.id) + ".npz", sock=self.sock,
                buffer_size=self.buffer_size, timeout=self.recv_timeout)
            self.grads_msg_size = fsize
        else:
            socket_recv(file="./tmp/grads_" + str(self.id) + ".npz", sock=self.sock,
                buffer_size=self.buffer_size, msg_size=self.grads_msg_size)
        #print("size of gradients file:", fsize)
        #print("Gradients received")

    def rv_wait(self):
        while True:
            # time window in which the server can use the lock
            time.sleep(0.5)
            self.lock.acquire()
            if self.ready_val.value == 1:
                self.release_lock()
                break
            self.release_lock()

    def run(self):
        print("Ready:", self.id)
        self.rv_wait()
        print("Start:", self.id)
        train_step = 0

        while self.work:
            self.send_update()
            if train_step % self.test_interval == 0:
                self.receive_test_results()
                # send to server process
                self.conn.send("recv_test")
                self.ready_val.value = 0
                self.rv_wait()

                self.send_update()
            self.receive_gradients()
            train_step += 1

            # send to server process
            self.conn.send("recv_train")

            self.ready_val.value = 0
            self.rv_wait()
        print("worker", self.id, "done")


class Server():
    def __init__(
            self,
            policy,
            model_dir,
            args_file,
            learning_rate=1e-3,
            global_norm_t=10,
            ip="127.0.0.1",
            port=5000,
            buffer_size=4096,
            n_clients=10,
            recv_timeout=4,
            test_interval=1000,
            train_p=0.8,
            seed=42,
            dataset="s3dis",
            p_data=1):
        mkdir("./tmp")
        s = socket.socket()
        s.bind((ip, port))
        s.listen(n_clients)

        c_id = 0
        self.pipes = []
        self.locks = []
        self.ready_vals = []
        self.processes = []
        self.polled = []
        self.n_clients = n_clients
        self.policy = policy
        self.model_dir = model_dir
        self.global_norm_t = global_norm_t
        self.test_interval = test_interval
        self.optimizer=tf.optimizers.Adam(learning_rate=learning_rate)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "./logs/siamese_supervised/" + current_time
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        self.train_step = 0

        self.policy.save(directory=self.model_dir, filename="tmp_net")

        print("Wait for clients")
        while c_id < n_clients:
            client_socket, address = s.accept()
            
            print("Transmit args to client {0}".format(c_id))
            msg = str(n_clients) + "," + str(c_id) + "," + str(test_interval) + "," + str(recv_timeout) + "," + str(train_p) + "," + str(seed) + "," + dataset + "," + args_file + "," + str(p_data)
            client_socket.send(msg.encode())
            print("Done - Wait for acknowledgement of client {0}".format(c_id))
            client_socket.recv(buffer_size)
            print("Acknowledgement received from client {0}".format(c_id))

            parent_conn, child_conn = Pipe(duplex=True)
            self.pipes.append(parent_conn)
            lock = Lock()
            self.locks.append(lock)
            rv = Value("i", 0)
            self.ready_vals.append(rv)
            cp = ClientProcess(conn=child_conn, pid=c_id, ready_val=rv, lock=lock,
                sock=client_socket, addr=address, model_dir=model_dir, buffer_size=buffer_size,
                recv_timeout=recv_timeout, test_interval=test_interval)
            self.processes.append(cp)
            cp.start()
            self.polled.append(False)
            print("Connected client nr {0}".format(c_id))
            c_id += 1
        self.grads = n_clients * [None]
        self.losses = n_clients * [None]
        self.tresults = n_clients * [None]

        print("All clients connected")
        time.sleep(3*len(self.processes))
        print("Unlock")
        for id in range(self.n_clients):
            self.unlock(id=id)

    def unlock(self, id):
        self.locks[id].acquire()
        self.ready_vals[id].value = 1
        self.locks[id].release()

    def store_grads(self, msg, id):
        #print("Store gradients")
        grad_data = np.load("./tmp/grads_" + str(id) + ".npz", allow_pickle=True)
        grad_data = dict(grad_data)
        tmp_grads = []
        grad_nrs = []
        loss_dict = {}

        vars_ = self.policy.get_vars(with_non_trainable=False)
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
        self.grads[id] = grads
        self.losses[id] = loss_dict

    def store_test_results(self, msg, id):
        test_data = np.load("./tmp/test_stats_" + str(id) + ".npz", allow_pickle=True)
        test_data = dict(test_data)
        self.tresults[id] = test_data

    def on_msg_received(self, msg, id):
        #print(msg, id)
        if self.test:
            self.store_test_results(msg=msg, id=id)
        else:
            self.store_grads(msg=msg, id=id)
        
    def avg_test_results(self):
        keys = list(self.tresults[0].keys())
        n_stats = len(keys)
        avg_stats = n_stats * [None]
        for i in range(n_stats):
            tmp_results = []
            for id in range(self.n_clients):
                test_data = self.tresults[id]
                key = keys[i]
                stat = test_data[key]
                tmp_results.append(stat)
            avg_results = np.average(tmp_results, axis=0)
            avg_stats[i] = avg_results
        return keys, avg_stats

    def avg_gradients(self):
        n_grads = len(self.grads[0])
        avg_grads = n_grads*[None]
        for i in range(n_grads):
            tmp_grads = []
            for id in range(self.n_clients):
                grads_id = self.grads[id]
                grad = grads_id[i]
                tmp_grads.append(grad)
            avg_grad = np.average(tmp_grads, axis=0)
            avg_grad = tf.convert_to_tensor(avg_grad, dtype=tf.float32)
            avg_grads[i] = avg_grad
        avg_grads = tuple(avg_grads)
        return avg_grads

    def write_train_results(self, global_norm):
        keys = list(self.losses[0].keys())
        n_losses = len(keys)
        with self.train_summary_writer.as_default():
            for i in range(n_losses):
                key = keys[i]
                tmp_losses = []
                for id in range(self.n_clients):
                    loss_dict = self.losses[id]
                    loss = loss_dict[key]
                    tmp_losses.append(loss)
                avg_loss = np.average(tmp_losses)
                tf.summary.scalar("train/avg_" + key, avg_loss, step=self.train_step)
            tf.summary.scalar("train/global_norm", global_norm, step=self.train_step)
        self.train_summary_writer.flush()

    def write_test_results(self):
        keys, avg_stats = self.avg_test_results()
        n_stats = len(keys)
        with self.train_summary_writer.as_default():
            for i in range(n_stats):
                key = keys[i]
                stat = avg_stats[i]
                tf.summary.scalar("test/" + key, stat, step=self.test_step)
        self.train_summary_writer.flush()

    def reduce(self):
        vars_ = self.policy.get_vars(with_non_trainable=False)
        grads = self.avg_gradients()
        if len(grads) != len(vars_):
            print("")
            for i in range(len(vars_)):
                print(vars_[i].name)
            raise Exception("Number mismatch. Gradients: {0}, Vars: {1}".format(len(grads), len(vars_)))

        global_norm = tf.linalg.global_norm(grads)
        if self.global_norm_t > 0:
            grads, _ = tf.clip_by_global_norm(
                grads,
                self.global_norm_t,
                use_norm=global_norm)
        for i in range(len(vars_)):
            # print(vars_[i].name, grads[i])
            if grads[i].shape != vars_[i].shape:
                raise Exception("Shape mismatch at variable {0}. A Shape: {1}, B Shape: {2}".format(vars_[i].name, grads[i].shape, vars_[i].shape))
        self.optimizer.apply_gradients(zip(grads, vars_))

        self.write_train_results(global_norm=global_norm)

    def msg_to_workers(self, msg):
        """Message to client processes/workers
        """
        for id in range(len(self.pipes)):
            self.pipes[id].send(msg)
    
    def recv_data(self, timeout=None):
        """Receive data from a process/worker
        """
        for id in range(self.n_clients):
            if not self.polled[id]:
                if self.pipes[id].poll(timeout=timeout):
                    msg = self.pipes[id].recv()
                    self.on_msg_received(msg, id)
                    self.polled[id] = True

    def start(self):
        """Main training loop
        """
        self.train_step = 0
        self.test_step = 0
        self.test = True
        while True:
            self.test = self.train_step % self.test_interval == 0
            #print("Test: {0}".format(self.test))
            self.recv_data()
            if self.test:
                self.write_test_results()
                self.test_step += 1
                for id in range(self.n_clients):
                    self.polled[id] = False
                    self.unlock(id=id)
                #print("Receive gradient data")
                self.test = False
                self.recv_data()
            self.reduce()
            self.policy.save(directory=self.model_dir, filename="tmp_net")
            self.train_step += 1
            for id in range(self.n_clients):
                self.polled[id] = False
                self.unlock(id=id)
        print("stop master loop")
        self.recv_data()
        time.sleep(3)
        self.msg_to_workers("stop")
        time.sleep(3)
        for id in range(self.n_clients):
            self.unlock(id=id)
        self.stop()