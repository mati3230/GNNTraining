from multiprocessing import Process
import time


class BaseWorkerProcess(Process):
    """Class that abstracts the communication to the master process.

    Parameters
    ----------
    conn : multiprocessing.Connection
        Connection to communicate to the master process.
    id : int
        ID of the worker (n cpu).
    ready_val : multiprocessing.Value
        Boolean shared value that indicates if the master process has processed
        the data from the workers.
    lock : multiprocessing.Lock
        A lock to for the ready_val.

    Attributes
    ----------
    work : boolean
        Status variable that keeps the worker working.
    conn : multiprocessing.Connection
        Connection to communicate to the master process.
    id : int
        ID of the worker (n cpu).
    ready_val : multiprocessing.Value
        Boolean shared value that indicates if the master process has processed
        the data from the workers.
    lock : multiprocessing.Lock
        A lock to for the ready_val.

    """
    def __init__(self, conn, id, ready_val, lock, start_with_work=True):
        """Constructor.

        Parameters
        ----------
        conn : multiprocessing.Connection
            Connection to communicate to the master process.
        id : int
            ID of the worker (n cpu).
        ready_val : multiprocessing.Value
            Boolean shared value that indicates if the master process has processed
            the data from the workers.
        lock : multiprocessing.Lock
            A lock to for the ready_val.
        """
        super().__init__()
        self.conn = conn
        self.id = id
        self.ready_val = ready_val
        self.lock = lock
        self.work = True
        self.start_with_work = start_with_work
        print("init:", self.id)

    def on_master_progress(self, msg):
        """Callback method after the master process has done its work.

        Parameters
        ----------
        msg : sring
            Instruction from the master process
        """
        pass

    def progress(self):
        """Method in which the worker can do its work."""
        pass

    def on_worker_start(self):
        """Callback method to preprare before the worker loop begins."""
        pass

    def check_master_progress(self):
        """Wait till the master sends new instructions."""
        if self.conn.poll(timeout=None):
            try:
                msg = self.conn.recv()
            except EOFError:
                print("EOFError occured")
                msg = "stop"
            print("check master process", msg)
            if msg == "stop":
                self.work = False
                return
            self.on_master_progress(msg)

    def release_lock(self):
        time.sleep(0.2)
        self.lock.release()

    def run(self):
        """Main method of the worker."""
        self.on_worker_start()
        print("ready:", self.id)
        if self.start_with_work:
            while(True):
                time.sleep(0.5)
                self.lock.acquire()
                if self.ready_val.value == 1:
                    self.release_lock()
                    break
                self.release_lock()
            print("start:", self.id)
            while self.work:
                progress = self.progress()
                self.conn.send(progress)
                #print(self.id, ": progress send")
                self.ready_val.value = 0
                while True:
                    time.sleep(0.5)
                    self.lock.acquire()
                    if self.ready_val.value == 1:
                        self.release_lock()
                        break
                    self.release_lock()
                self.ready_val.value = 0
                #print(self.id, ": check master - reset ready val")
                self.check_master_progress()
        else:
            print("start:", self.id)
            while self.work:
                self.ready_val.value = 0
                while True:
                    time.sleep(0.5)
                    self.lock.acquire()
                    if self.ready_val.value == 1:
                        self.release_lock()
                        break
                    self.release_lock()
                self.ready_val.value = 0
                #print("check cmd", self.id)
                self.check_master_progress()
                progress = self.progress()
                self.conn.send(progress)
                #print(self.id, ": progress send")
                #print(self.id, ": check master - reset ready val")


        print("worker", self.id, "done")
