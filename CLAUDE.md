# CLAUDE.md

- If a task is estimated to take **over ~2 hours** (e.g. a GPU retrain, a long multi-run sweep), **stop and ask the owner before starting it.**

- **Hardware:** this Mac = M3 Ultra (32 CPU cores, 80-core GPU, 256 GB); threebody = RTX 4090 (24 GB) runs Qwen. Parallelize across the cores, but the single Mac GPU is the bottleneck — share MLX model servers (`--shared-text-laya`), don't spawn one per worker.

- **Only Qwen may go to threebody — nothing else; all laya (eye + text-laya) runs on the Mac.**

- **Worker counts: TRAIN with 8 workers, MEASURE/TEST with 16.** Training is GPU-bound on the shared text-laya server (every decision calls laya), so 16 workers just serialize on the GPU — measured: 16w gave 2× the games for ~1.6× the wall-time, no speedup. Measuring is lighter (no Qwen, no learning, no explore) and parallelizes fine across the M3's cores, so 16 workers there is a real speedup.
