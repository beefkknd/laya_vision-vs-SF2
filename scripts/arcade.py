"""The arcade console: N Mesen windows, laya playing Chun-Li in each, forever; every game's decisions tailed here in
its own color.

    python scripts/arcade.py                       # 3 games, runs/r1/best
    python scripts/arcade.py --games 2 --model runs/r2/best

Each game is one scripts/show.py process (its own model copy and windowed Mesen on port base+i), so the games run
side by side on the GPU instead of taking turns. Ctrl-C stops every game and closes its Mesen.
"""
import argparse
import os
import signal
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = ["\033[96m", "\033[93m", "\033[95m", "\033[92m", "\033[94m", "\033[91m"]  # cyan yellow magenta green ...
RESET = "\033[0m"


def tail(i: int, proc: subprocess.Popen, lock: threading.Lock) -> None:
    color = COLORS[i % len(COLORS)]
    for line in proc.stdout:
        with lock:
            sys.stdout.write("%s[game %d] %s%s\n" % (color, i + 1, line.rstrip(), RESET))
            sys.stdout.flush()
    with lock:
        sys.stdout.write("%s[game %d] exited (%s)%s\n" % (color, i + 1, proc.wait(), RESET))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--model", default="runs/r1/best")
    ap.add_argument("--base-port", type=int, default=47801)
    args, passthrough = ap.parse_known_args()  # anything else goes to every show.py (e.g. --headless)

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGHUP, stop)  # closing the console window stops the games too
    signal.signal(signal.SIGTERM, stop)

    procs, lock = [], threading.Lock()
    for i in range(args.games):
        argv = [sys.executable, "-u", os.path.join(HERE, "show.py"), "--model", args.model,
                "--port", str(args.base_port + i), "--seed", str(i),
                "--jitter-base", str(i * 30), *passthrough]  # each game idles differently: different fights
        procs.append(subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      start_new_session=True))  # our Ctrl-C is relayed below, once
    threads = [threading.Thread(target=tail, args=(i, p, lock), daemon=True) for i, p in enumerate(procs)]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nstopping %d games..." % len(procs), flush=True)
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)  # show.py closes its Mesen on the way out
        for p in procs:
            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
