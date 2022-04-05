import math
from datetime import datetime


def divide_work(worker_id, n_workers, workload):
    work = math.floor(workload / n_workers)
    if worker_id == n_workers - 1:
        work += workload % n_workers
    start_w = work * worker_id
    stop_w = start_w + work
    
    return work, start_w, stop_w


def divide_work_equally(n_workers, workload):
    min_wl = math.floor(workload / n_workers)
    work_packages = {}
    for i in range(n_workers):
        start = i*min_wl
        stop = start + min_wl
        work_packages[i] = list(range(start, stop))
    remaining = workload % n_workers
    last_id = stop
    i = 0
    for i in range(remaining):
        work_packages[i].append(last_id + i)
        i += 1
        if i == n_workers:
            i = 0


def process_range(workload, n_cpus, process_class, target, args):
    intervals = math.floor(workload / n_cpus)
    processes = []
    for i in range(n_cpus):
        min_i = i*intervals
        max_i = min_i + intervals
        if i == n_cpus - 1:
            max_i = workload 
        p = process_class(target=target, args=(i, args, min_i, max_i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def print_progress(pid, i, workload):
    time = datetime.now().strftime("%H:%M:%S")
    print("PID {0}, {1:.2f}%, {2}".format(pid, ((i+1)/workload)*100, time))