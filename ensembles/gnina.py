#!/usr/bin/env python3
import os
import time
import pandas as pd
import subprocess
from datetime import datetime

# ========= CONFIG (edit these) ===============================================
PICKLE_PATH = "dldocking/gnina_dockings.pkl"   # <- path to the pickle with the list
MAX_PARALLEL = 2                       # how many to run at once
# Set a cutoff like "2025-11-05 23:30" (local to TIMEZONE); or set to None to disable cutoff
CUTOFF_STR = '2025-11-12 08:00:00'
POLL_INTERVAL_SEC = 1.0                   # how often to check for finished procs
# ============================================================================


def parse_cutoff(cutoff_str: str | None) -> datetime | None:
    if not cutoff_str:
        return None
    fmt = "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(cutoff_str, fmt)


def load_tasks(pickle_path: str):
    """
    Expect the pickle to contain an iterable of dicts with keys:
      - 'cmd': str (the shell command to run)
      - 'path': str or Path (working directory)
    """
    obj = pd.read_pickle(pickle_path)
    # Convert to a plain list of dicts with 'cmd' and 'path'
    tasks = []
    for item in obj:
        cmd = item["cmd"]
        path = item["path"]
        path = str(path)  # ensure str for cwd
        tasks.append({"cmd": cmd, "path": path})
    return tasks

def open_log(path: str):
    os.makedirs(path, exist_ok=True)
    logfile = os.path.join(path, "log.txt")
    # overwrite each run, same as original sample
    return open(logfile, "w", buffering=1)  # line-buffered

def start_task(task):
    """
    Start a single task with Popen, capturing stdout+stderr to log.txt,
    and return a dict representing the running job.
    """
    f = open_log(task["path"])
    f.write(f"Command: {task['cmd']}\n")
    f.write(f"Started: {datetime.now().isoformat()}\n\n")
    f.flush()
    # shell=True to allow inline env assignments like LD_LIBRARY_PATH=...
    proc = subprocess.Popen(
        task["cmd"],
        shell=True,
        cwd=task["path"],
        stdout=f,
        stderr=f,
        text=True,
    )
    return {"proc": proc, "fh": f, "start": time.time(), "task": task}

def finalize_job(job):
    """
    Wait is done already when we call this (or poll indicated finished).
    Append elapsed time and returncode into the log, close file.
    """
    end = time.time()
    h, rem = divmod(end - job["start"], 3600)
    m, s = divmod(rem, 60)
    rc = job["proc"].returncode
    fh = job["fh"]
    fh.write(f"\nTime elapsed: {int(h):02d}:{int(m):02d}:{s:05.2f}\n")
    fh.write(f"Return code: {rc}\n")
    fh.close()
    return rc

def main():
    cutoff_dt = parse_cutoff(CUTOFF_STR)

    tasks = load_tasks(PICKLE_PATH)
    task_iter = iter(tasks)
    running = []
    submitted = 0
    completed = 0
    failed = 0

    def before_cutoff() -> bool:
        if cutoff_dt is None:
            return True
        return datetime.now() < cutoff_dt

    # Fill initial slots up to MAX_PARALLEL (obeying cutoff)
    while len(running) < MAX_PARALLEL and before_cutoff():
        try:
            next_task = next(task_iter)
        except StopIteration:
            break
        running.append(start_task(next_task))
        submitted += 1
        print(f"Submitted {submitted}: {next_task['path']}")

    # Main loop: when any finishes, start a new one (if before cutoff)
    try:
        while running:
            # Check which processes finished
            still_running = []
            for job in running:
                rc = job["proc"].poll()
                if rc is None:
                    still_running.append(job)
                else:
                    completed += 1
                    if rc != 0:
                        failed += 1
                    finalize_job(job)
                    print(f"Finished {completed} (rc={rc}): {job['task']['path']}")

            running = still_running

            # Try to top up the slots if we are still before cutoff
            while len(running) < MAX_PARALLEL and before_cutoff():
                try:
                    next_task = next(task_iter)
                except StopIteration:
                    break
                running.append(start_task(next_task))
                submitted += 1
                print(f"Submitted {submitted}: {next_task['path']}")

            # If we hit cutoff, we won't submit more; just wait for current to finish
            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("KeyboardInterrupt received; no new tasks will be submitted. Waiting for running tasks to finish...")
        # Do not kill running tasks—let them finish
        while running:
            still_running = []
            for job in running:
                rc = job["proc"].poll()
                if rc is None:
                    still_running.append(job)
                else:
                    completed += 1
                    if rc != 0:
                        failed += 1
                    finalize_job(job)
                    print(f"Finished {completed} (rc={rc}): {job['task']['path']}")
            running = still_running
            time.sleep(POLL_INTERVAL_SEC)

    # Summary
    print(f"Submitted: {submitted}, Completed: {completed}, Failed: {failed}")
    # If cutoff prevented submitting some tasks, let the user know at stdout
    if cutoff_dt is not None and submitted < len(tasks):
        remaining = len(tasks) - submitted
        print(f"Cutoff reached; {remaining} tasks were not started.")

if __name__ == "__main__":
    main()


