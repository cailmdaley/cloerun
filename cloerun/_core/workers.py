"""Threads and cores of the likelihood processes.

With ``sampler.threads: n``, every process that evaluates the likelihood runs XLA, OpenMP and
the BLAS libraries on ``n`` threads, and the k-th worker of the sampler's pool is pinned to its
own ``n`` cores: on a node, ``pool: [cores / n, ...]`` with ``threads: 1`` runs one
single-threaded likelihood per core, which for a compiled JAX likelihood is the fastest layout.
The pool's workers are spawned, not forked (a process forked after JAX or CAMB has started its
threads can deadlock), and read the thread counts from the environment as they start.
"""

from __future__ import annotations

import multiprocessing
import os


def use_threads(threads: int | None) -> None:
    """Set the thread counts of this process and of the workers it spawns; call before the
    first JAX computation. ``None`` leaves the libraries' defaults."""
    if threads is None:
        return
    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen={str(threads > 1).lower()} "
        f"intra_op_parallelism_threads={threads}"
    )
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)
    multiprocessing.set_start_method("spawn", force=True)


def pin(threads: int | None) -> None:
    """Pin the k-th pool worker, every thread of it, to its own ``threads`` cores (no-op outside
    a pool)."""
    worker = multiprocessing.current_process()._identity
    if threads is None or not worker:
        return
    cores = sorted(os.sched_getaffinity(0))
    first = (worker[-1] - 1) * threads % len(cores)
    # Every thread of the process: those JAX started at import (its compute pool among them)
    # keep the affinity they were created with.
    for thread in os.listdir("/proc/self/task"):
        os.sched_setaffinity(int(thread), cores[first:first + threads])
