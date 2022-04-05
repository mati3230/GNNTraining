from abc import ABC, abstractmethod
import multiprocessing as mp
import threading
import time
import psutil

class BaseMasterProcess(ABC):
    """Class that abstracts the master process loop and the communication with
    the worker processes.

    Parameters
    ----------
    n_cpus : int
        Number of workers.
    shared_value : mp.Value
        Multiprocessing value to stop the training.
    critical_mem : int
        Critical value of acquired RAM.

    Attributes
    ----------
    processes : dict
        Key: process id
        Value: a worker process
    pipes : dict
        Key: process id
        Value: mp.Pipe to communicate with the worker processes
    polled : dict
        Key: process id
        Value: mp.Pipe to communicate with the worker processes
    ready_vals : list(mp.Value)
        Description of attribute `ready_vals`.
    locks : list(mp.Lock)
        Description of attribute `locks`.
    n_cpus : int
        Number of workers.
    shared_value : mp.Value
        Multiprocessing value to stop the training.
    critical_mem : int
        Critical value of acquired RAM.

    """
    def __init__(self, n_cpus, shared_value, critical_mem, spawn=True):
        """Constructor.

        Parameters
        ----------
        n_cpus : int
            Number of workers.
        shared_value : mp.Value
            Multiprocessing value to stop the training.
        critical_mem : int
            Critical value of acquired RAM.
        """
        super().__init__()
        if spawn:
            mp.set_start_method("spawn", force=True)
        self.n_cpus = n_cpus
        self.shared_value = shared_value
        self.processes = {}
        self.pipes = {}
        self.polled = {}
        self.ready_vals = []
        self.locks = []
        self.critical_mem = critical_mem

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

    def msg_to_workers(self, msg):
        """Send an instruction to the workers.

        Parameters
        ----------
        msg : str
            The instruction.

        """
        #print("send msg:", msg)
        for id in range(len(self.pipes)):
            self.pipes[id].send(msg)

    def stop(self):
        """Stops the workers."""
        print("stop")
        for id in range(len(self.processes)):
            p = self.processes[id]
            #p.terminate()
            p.join()
            self.pipes[id].close()
        self.pipes.clear()
        print("everything stopped")

    def create_connections(self):
        """Creates the worker, connections to the workers."""
        # create processes
        for id in range(self.n_cpus):
            parent_conn, child_conn = mp.Pipe(duplex=True)
            self.pipes[id] = parent_conn
            rv = mp.Value("i", 0)
            self.ready_vals.append(rv)
            lock = mp.Lock()
            self.locks.append(lock)
            p = self.create_worker(child_conn, id, rv, lock)
            p.start()
            self.processes[id] = p
            self.polled[id] = False
        time.sleep(3*len(self.processes))
        print("unlock")
        for id in range(self.n_cpus):
            self.locks[id].acquire()
            self.ready_vals[id].value = 1
            self.locks[id].release()

    def recv_data(self, timeout=None):
        """Receive data from the worker process.

        Parameters
        ----------
        timeout : int
            Seconds to call a time timeout. If None, this method will block
            until some data is received.

        Returns
        -------
        boolean
            False, if any error is appeared while receiving the data.
        """
        #for id in range(self.n_cpus):
        #    self.ready_vals[id].value = 0
        for id in range(self.n_cpus):
            if not self.polled[id]:
                if self.pipes[id].poll(timeout=timeout):
                    msg = self.pipes[id].recv()
                    self.on_worker_progress(msg, id)
                    self.polled[id] = True
                else:
                    return False
        return True

    def start_loop(self):
        """Method to start the loop (training, testing, ...). Main master
        loop."""
        print("create connections")
        self.create_connections()
        print("connections created")
        while self.shared_value.value:
            #print("master: recv data")
            self.recv_data()
            #print("master: start loop")
            self.on_loop()
            #print("master: loop finished")
            self.on_loop_end()
            #time.sleep(1)
            for id in range(self.n_cpus):
                self.polled[id] = False
                self.locks[id].acquire()
                self.ready_vals[id].value = 1
                self.locks[id].release()
            # workers should need more time than this sleep time
            # print("master: ready vals unlocked - sleep")
            #time.sleep(2)
            #print("master: done")
            # check memory
            mem = psutil.virtual_memory()
            if mem.percent > self.critical_mem:
                print("Out of memory. More than " + str(self.critical_mem) + "% of your memory are in use.")
                self.shared_value.value = False
        print("stop master loop")
        self.recv_data()
        time.sleep(3)
        self.msg_to_workers("stop")
        time.sleep(3)
        for id in range(self.n_cpus):
            self.locks[id].acquire()
            self.ready_vals[id].value = 1
            self.locks[id].release()
        self.stop()
