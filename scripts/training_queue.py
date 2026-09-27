"""Resource-aware queue for one-GPU training.

    python scripts/training_queue.py create --train-out runs/chunli_r5 --train-init runs/chunli_r4/best \\
      --train-data data/<set_a> --train-data data/<set_b>
    python scripts/training_queue.py run

The pool has one GPU slot (a hand-started train.py holds it too) and six CPU slots. At the deadline it stops
running queue jobs and moves pending work to history; without ``--minutes`` there is no deadline and training runs
its full epoch (or early-stops).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUEUE = ROOT / "out" / "training_queue.json"
LOG = ROOT / "out" / "training.log"  # one stable file for tail -f: task output and queue events
CPU_SLOTS, GPU_SLOTS = 6, 1
DEADLINE_MARGIN_S = 60  # train.py wraps up (final eval, train_log.json) this long before the queue kills it


def load(path):
    with open(path) as f:
        return json.load(f)


def save(path, queue):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(queue, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def live_external_gpu():
    """A hand-started trainer reserves the sole GPU slot too."""
    out = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True, check=True).stdout
    me = str(os.getpid())
    return any("scripts/train.py" in line and not line.lstrip().startswith(me + " ") for line in out.splitlines())


def command(task, remaining):
    kind, spec = task["kind"], task["spec"]
    if kind == "train":
        out = ROOT / spec["out"]
        if out.exists():
            raise RuntimeError("refusing existing train output: %s" % out)
        argv = [sys.executable, "scripts/train.py"]
        for data in spec["data"]:
            argv += ["--data", data]
        argv += ["--init", spec["init"], "--out", spec["out"], "--epochs", "1", "--batch-size", "8", "--patience", "3"]
        if remaining is not None:  # no deadline: train runs its epoch (or early-stops)
            argv += ["--max-minutes", "%.2f" % ((remaining - DEADLINE_MARGIN_S) / 60)]
        return argv
    raise RuntimeError("unknown task kind: %s" % kind)


def note(msg):
    print(msg, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write("=== %s %s ===\n" % (time.strftime("%H:%M:%S"), msg))


def dependencies_done(queue, task):
    states = {t["id"]: t["state"] for t in queue["tasks"]}
    return all(states.get(dep) == "done" for dep in task.get("depends", []))


def expire(queue):
    for task in queue["tasks"]:
        if task["state"] == "pending":
            task["state"] = "expired"
            queue["history"].append(task.copy())
    # History remains inspectable; no task remains claimable after the deadline.
    queue["tasks"] = []
    queue["state"] = "expired"


def run(path, poll_seconds):
    queue = load(path)
    running = {}
    logs = {}
    while True:
        now = time.time()
        if queue["deadline"] is not None and now >= queue["deadline"]:
            for task_id, proc in running.items():
                os.killpg(proc.pid, signal.SIGTERM)
                task = next(t for t in queue["tasks"] if t["id"] == task_id)
                task["state"] = "timed_out"
                task["finished_at"] = now
                queue["history"].append(task.copy())
            for f in logs.values():
                f.close()
            expire(queue)
            save(path, queue)
            note("deadline reached; queue cleared")
            return

        for task_id, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            logs.pop(task_id).close()
            task = next(t for t in queue["tasks"] if t["id"] == task_id)
            task["state"] = "done" if rc == 0 else "failed"
            task["returncode"], task["finished_at"] = rc, time.time()
            queue["history"].append(task.copy())
            running.pop(task_id)
            save(path, queue)
            note("%s %s (exit %d)" % (task_id, task["state"], rc))

        used_cpu = sum(next(t for t in queue["tasks"] if t["id"] == i)["slots"] for i in running)
        used_gpu = sum(1 for i in running if next(t for t in queue["tasks"] if t["id"] == i)["resource"] == "gpu")
        for task in queue["tasks"]:
            if task["state"] != "pending" or not dependencies_done(queue, task):
                continue
            if task["resource"] == "cpu" and used_cpu + task["slots"] > CPU_SLOTS:
                continue
            if task["resource"] == "gpu" and (used_gpu >= GPU_SLOTS or live_external_gpu()):
                continue
            try:
                argv = command(task, None if queue["deadline"] is None else queue["deadline"] - now)
            except RuntimeError as e:
                task["state"], task["error"] = "failed", str(e)
                queue["history"].append(task.copy())
                save(path, queue)
                note("%s failed to start: %s" % (task["id"], e))
                continue
            note("started %s: %s" % (task["id"], " ".join(argv)))
            log = open(LOG, "a")
            proc = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            task["state"], task["pid"], task["started_at"] = "running", proc.pid, now
            running[task["id"]], logs[task["id"]] = proc, log
            used_cpu += task["slots"] if task["resource"] == "cpu" else 0
            used_gpu += 1 if task["resource"] == "gpu" else 0
            save(path, queue)

        if not running and not any(t["state"] == "pending" for t in queue["tasks"]):
            queue["state"] = "done"
            save(path, queue)
            note("queue complete")
            return
        time.sleep(poll_seconds)


def create(args):
    path = Path(args.queue)
    if path.exists():
        raise SystemExit("queue exists: %s" % path)
    queue = {
        "version": 1,
        "state": "running",
        "created_at": time.time(),
        "deadline": None if args.minutes is None else time.time() + args.minutes * 60,
        "pool": {"gpu": GPU_SLOTS, "cpu": CPU_SLOTS},
        "history": [],
        "tasks": [
            {"id": "train", "kind": "train", "resource": "gpu", "slots": 1, "state": "pending",
             "spec": {"out": args.train_out, "init": args.train_init, "data": args.train_data}},
        ],
    }
    save(path, queue)
    print("created %s: 1 GPU training slot, %s"
          % (path, "no deadline" if args.minutes is None else "%.0f-minute deadline" % args.minutes))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create")
    c.add_argument("--queue", default=str(DEFAULT_QUEUE))
    c.add_argument("--minutes", type=float, default=None, help="queue deadline; default none (train runs its epoch)")
    c.add_argument("--train-out", required=True)
    c.add_argument("--train-init", required=True)
    c.add_argument("--train-data", action="append", required=True)
    r = sub.add_parser("run")
    r.add_argument("--queue", default=str(DEFAULT_QUEUE))
    r.add_argument("--poll-seconds", type=float, default=5)
    args = ap.parse_args()
    if args.command == "create":
        create(args)
    else:
        run(Path(args.queue), args.poll_seconds)


if __name__ == "__main__":
    main()
