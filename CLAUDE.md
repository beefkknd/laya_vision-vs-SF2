# CLAUDE.md

- If a task is estimated to take **over ~2 hours** (e.g. a GPU retrain, a long multi-run sweep), **stop and ask the owner before starting it.**

- **Hardware:** this Mac = M3 Ultra (32 CPU cores, 80-core GPU, 256 GB); threebody = RTX 4090 (24 GB) runs Qwen. Parallelize across the cores, but the single Mac GPU is the bottleneck — share MLX model servers (`--shared-text-laya`), don't spawn one per worker.

- **Only Qwen may go to threebody — nothing else; all laya (eye + text-laya) runs on the Mac.**
