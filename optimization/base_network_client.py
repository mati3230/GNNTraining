import socket
from abc import abstractmethod

from environment.utils import mkdir
from .base_master_process import BaseMasterProcess

class Client(BaseMasterProcess):
    def __init__(
            self,
            n_cpus,
            shared_value,
            critical_mem,
            server_ip,
            server_port,
            init_buffer=4096,
            data_dir="./tmp",
            spawn=True):
        # conn, id, ready_val, lock, start_with_work=True
        super().__init__(n_cpus=n_cpus, shared_value=shared_value, critical_mem=critical_mem, spawn=spawn)
        self.data_dir = data_dir
        mkdir(data_dir)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((server_ip, server_port))
        print("wait for args data from the server")
        msg = self.sock.recv(init_buffer)
        print("args received")

        msg = msg.decode()
        msg = msg.split(",")

        self.n_nodes = int(msg[0])
        self.node_id = int(msg[1])
        self.recv_timeout = int(msg[2])
        print("n_clients: {0}, client_id: {1}".format(self.n_nodes, self.node_id))
        print("recv_timeout: {0}".format(self.recv_timeout))
        self.unpack_msg(msg=msg, i=2)

        self.sock.send(str(n_cpus).encode())
        print("acknowledgement send")
        self.n_cpus = n_cpus
        
        self.on_init()

    @abstractmethod
    def create_worker(self, conn, id, ready_val, lock):
        """Creates a worker process.

        Parameters
        ----------
        conn : multiprocessing.Connection
            Connection to communicate to the master process.
        id : int
            ID of the worker (n cpu).
        ready_val : multiprocessing.Value
            Flag to indicate if the master process is ready to receive data.
        lock : multiprocessing.Lock
            Lock for the ready_val

        Returns
        -------
        BaseWokerProcess
            Instance of a BaseWokerProcess.

        """
        pass

    @abstractmethod
    def on_worker_progress(self, msg, id):
        """Callback method when a worker finished its working loop.

        Parameters
        ----------
        msg : list
            List with information from the worker.
        id : int
            ID of the worker (n cpu).
        """
        pass

    @abstractmethod
    def on_loop(self):
        """Implementation of the master work."""
        pass

    @abstractmethod
    def on_loop_end(self):
        """Send next instruction to the worker."""
        pass

    @abstractmethod
    def unpack_msg(self, msg):
        return

    @abstractmethod
    def on_init(self):
        return