"""Resource-aware queue for headless collection and one-GPU training.

Pipelined rounds: train round N on data already on disk while the CPU pool
collects round N+1's data, then run the pool:

    python scripts/training_queue.py create --minutes 30 --collect-name seed_chunli_r5 \\
      --savestate states/<fight>.state --rom "$SF2_ROM" \\
      --train-out runs/chunli_r5 --train-init runs/chunli_r4/best \\
      --train-data data/seed_chunli_r4 --train-data data/dagger_chunli_r2
    python scripts/training_queue.py run

Training waits for collection only when ``--train-data`` names the collection's
own ``data/<collect-name>``; otherwise both start at once.

The pool has one GPU slot and six CPU slots. Collection occupies all six CPU
slots through ``parallel.py``; training occupies the single GPU slot (plus its
data-loader processes on the remaining cores). At the deadline it stops running
queue jobs and moves pending work to history.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import _path  # noqa: F401
from sf2.headless import find_mesen

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUEUE = ROOT / "out" / "training_queue.json"
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
    if kind == "collect":
        return [sys.executable, "scripts/parallel.py", "--workers", str(spec["workers"]), "--base-port",
                str(spec["base_port"]), "collect_teacher", "--name", spec["name"], "--decisions",
                str(spec["decisions"]), "--eps", str(spec["eps"]), "--savestate", spec["savestate"],
                "--me", spec["me"], "--opp", spec["opp"], "--rom", spec["rom"], "--mesen", spec["mesen"]]
    if kind == "train":
        out = ROOT / spec["out"]
        if out.exists():
            raise RuntimeError("refusing existing train output: %s" % out)
        argv = [sys.executable, "scripts/train.py"]
        for data in spec["data"]:
            argv += ["--data", data]
        argv += ["--init", spec["init"], "--out", spec["out"], "--epochs", "1", "--batch-size", "8",
                 "--patience", "3", "--max-minutes", "%.2f" % ((remaining - DEADLINE_MARGIN_S) / 60)]
        return argv
    raise RuntimeError("unknown task kind: %s" % kind)


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
        if now >= queue["deadline"]:
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
            print("deadline reached; queue cleared")
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
            print("%s %s" % (task_id, task["state"]), flush=True)

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
                argv = command(task, queue["deadline"] - now)
            except RuntimeError as e:
                task["state"], task["error"] = "failed", str(e)
                queue["history"].append(task.copy())
                save(path, queue)
                continue
            # One stable path makes live monitoring independent of task names.
            log_path = ROOT / "out" / "training.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(log_path, "a")
            log.write("\n=== %s started %s ===\n" % (task["id"], time.strftime("%Y-%m-%d %H:%M:%S")))
            log.flush()
            proc = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            task["state"], task["pid"], task["started_at"] = "running", proc.pid, now
            running[task["id"]], logs[task["id"]] = proc, log
            used_cpu += task["slots"] if task["resource"] == "cpu" else 0
            used_gpu += 1 if task["resource"] == "gpu" else 0
            save(path, queue)
            print("started %s: %s" % (task["id"], " ".join(argv)), flush=True)

        if not running and not any(t["state"] == "pending" for t in queue["tasks"]):
            queue["state"] = "done"
            save(path, queue)
            print("queue complete")
            return
        time.sleep(poll_seconds)


def create(args):
    path = Path(args.queue)
    if path.exists():
        raise SystemExit("queue exists: %s" % path)
    rom = args.rom or os.environ.get("SF2_ROM")
    if not rom or not os.path.isfile(rom):
        # checked here, not 6 workers into the run
        raise SystemExit("ROM not found: pass --rom or set SF2_ROM (got %r)" % rom)
    try:
        mesen = find_mesen(args.mesen)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    collected = os.path.normpath(os.path.join("data", args.collect_name))
    waits = collected in {os.path.normpath(d) for d in args.train_data}
    queue = {
        "version": 1,
        "state": "running",
        "created_at": time.time(),
        "deadline": time.time() + args.minutes * 60,
        "pool": {"gpu": GPU_SLOTS, "cpu": CPU_SLOTS},
        "history": [],
        "tasks": [
            {"id": "collect", "kind": "collect", "resource": "cpu", "slots": CPU_SLOTS, "state": "pending",
             "spec": {"name": args.collect_name, "decisions": args.collect_decisions, "workers": CPU_SLOTS,
                      "base_port": args.base_port, "eps": args.eps, "savestate": args.savestate,
                      "me": args.me, "opp": args.opp, "rom": os.path.abspath(rom), "mesen": mesen}},
            {"id": "train", "kind": "train", "resource": "gpu", "slots": 1, "state": "pending",
             "depends": ["collect"] if waits else [], "spec": {"out": args.train_out, "init": args.train_init,
                                                    "data": args.train_data}},
        ],
    }
    save(path, queue)
    print("created %s: 6 CPU collection slots + 1 GPU training slot, %.0f-minute deadline, train %s"
          % (path, args.minutes, "after collect" if waits else "alongside collect"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create")
    c.add_argument("--queue", default=str(DEFAULT_QUEUE))
    c.add_argument("--minutes", type=float, default=30)
    c.add_argument("--collect-name", required=True)
    c.add_argument("--collect-decisions", type=int, default=18000)
    c.add_argument("--base-port", type=int, default=47940)
    c.add_argument("--eps", type=float, default=0.20)
    c.add_argument("--savestate", required=True)
    c.add_argument("--me", default="chunli")
    c.add_argument("--opp", default="dhalsim")
    c.add_argument("--rom", default=None, help="default $SF2_ROM; stored in the queue so workers need no env")
    c.add_argument("--mesen", default=None, help="default $SF2_MESEN, then the usual app locations")
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
