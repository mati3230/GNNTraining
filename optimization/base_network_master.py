import socket
import time
from multiprocessing import Pipe, Process, Lock, Value
from abc import ABC, abstractmethod


class NodeProcess(Process):
    """
    Network communication with the nodes
    """
    def __init__(self, conn, pid, ready_val, lock, sock, addr, buffer_size, recv_timeout, n_node_cpus):
        super().__init__()
        self.conn = conn
        self.id = pid
        self.ready_val = ready_val
        self.lock = lock
        self.sock = sock
        self.addr = addr
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout
        print("Init:", self.id)
        self.work = True
        self.n_node_cpus = n_node_cpus

    def release_lock(self):
        time.sleep(0.2)
        self.lock.release()

    @abstractmethod
    def send_update(self):
        pass

    def rv_wait(self):
        while True:
            # time window in which the server can use the lock
            time.sleep(0.5)
            self.lock.acquire()
            if self.ready_val.value == 1:
                self.release_lock()
                break
            self.release_lock()

    @abstractmethod
    def on_run_start(self):
        pass

    @abstractmethod
    def recv_loop(self):
        pass

    @abstractmethod
    def on_stop(self):
        pass

    def run(self):
        print("ready:", self.id)
        self.rv_wait()
        print("start:", self.id)
        self.on_run_start()

        while self.work:
            self.send_update()
            recv_msg = self.recv_loop()
            #print("msg to master: {0}".format(recv_msg))
            try:
                self.conn.send(recv_msg)
            except:
                break
            self.ready_val.value = 0
            self.rv_wait()
        print("stop:", self.id)
        self.on_stop()
        print("worker", self.id, "done")

class Server(ABC):
    def __init__(
            self,
            ip="127.0.0.1",
            port=5000,
            buffer_size=4096,
            n_nodes=10,
            recv_timeout=4
            ):
        self.work_loop = True
        self.on_init()
        # create server socket
        self.sock = socket.socket()
        self.sock.bind((ip, port))
        self.sock.listen(n_nodes)

        node_id = 0 # node id
        self.pipes = {}
        self.locks = []
        self.ready_vals = []
        self.processes = []
        self.polled = []
        self.n_nodes = n_nodes
        self.n_total_cpus = 0
        self.cpus_per_node = {}

        print("wait for nodes")
        while node_id < n_nodes:
            try:
                node_socket, address = self.sock.accept()
            except:
                self.stop(sock_only=True)
            
            print("transmit args to node {0}".format(node_id))
            add_msg = self.get_init_msg()
            msg = str(n_nodes) + "," + str(node_id) + "," + str(recv_timeout)
            if len(add_msg) > 0:
                msg += "," + add_msg
            node_socket.send(msg.encode())
            print("done - Wait for acknowledgement of node {0}".format(node_id))
            # node should transmit the number of cpus
            n_node_cpus = node_socket.recv(buffer_size)
            n_node_cpus = n_node_cpus.decode()
            n_node_cpus = int(n_node_cpus)
            self.cpus_per_node[node_id] = n_node_cpus
            self.n_total_cpus += n_node_cpus
            print("acknowledgement received from node {0}".format(node_id))

            parent_conn, child_conn = Pipe(duplex=True)
            self.pipes[node_id] = parent_conn
            lock = Lock()
            self.locks.append(lock)
            rv = Value("i", 0)
            self.ready_vals.append(rv)
            np = self.create_node_process(conn=child_conn, pid=node_id, ready_val=rv, lock=lock,
                sock=node_socket, addr=address, buffer_size=buffer_size, recv_timeout=recv_timeout, n_node_cpus=n_node_cpus)
            self.processes.append(np)
            np.start()
            self.polled.append(False)
            print("connected node nr {0}".format(node_id))
            node_id += 1
        
        self.on_node_start()

        print("all nodes connected. {0} CPUs will be used in the cluster.".format(self.n_total_cpus))
        time.sleep(3*len(self.processes))
        print("unlock") # start signal for the node process
        for id in range(self.n_nodes):
            self.unlock(id=id)

    @abstractmethod
    def on_init(self):
        pass

    @abstractmethod
    def create_node_process(self, conn, pid, ready_val, lock, sock, addr, buffer_size, recv_timeout, n_node_cpus):
        pass

    @abstractmethod
    def get_init_msg(self):
        return ""

    @abstractmethod
    def on_node_start(self):
        pass

    def unlock(self, id):
        self.locks[id].acquire()
        self.ready_vals[id].value = 1
        self.locks[id].release()

    def msg_to_workers(self, msg):
        """Message to node processes/workers
        """
        for id in range(len(self.pipes)):
            self.pipes[id].send(msg)
    
    @abstractmethod
    def on_msg_received(self, msg, id):
        pass

    def recv_data(self, timeout=None):
        """Receive data from a process/worker
        """
        #print("master: recv data")
        for id in range(self.n_nodes):
            if not self.polled[id]:
                if self.pipes[id].poll(timeout=timeout):
                    #print("master: wait for msg")
                    msg = self.pipes[id].recv()
                    self.on_msg_received(msg, id)
                    self.polled[id] = True
        #print("master: done")

    @abstractmethod
    def on_start(self):
        pass

    @abstractmethod
    def on_recv(self):
        pass

    @abstractmethod
    def on_loop(self):
        pass

    def stop(self, sock_only=False):
        if not sock_only:
            self.recv_data()
            time.sleep(3)
            self.msg_to_workers("stop")
            time.sleep(3)
            for id in range(self.n_nodes):
                self.unlock(id=id)
        self.sock.close()

    @abstractmethod
    def on_stop(self):
        pass

    def start(self):
        """Main training loop
        """
        self.on_start()

        while self.work_loop:
            self.on_recv()
            self.recv_data()
            self.on_loop()
            for id in range(self.n_nodes):
                self.polled[id] = False
                self.unlock(id=id)
        self.on_stop()
        print("stop master loop")
        self.stop()