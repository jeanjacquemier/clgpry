import json
import os
import socket
import platform
import sys
from datetime import datetime


def write_benchmark_file(run_dir, args, n_mpi, status, run_wall_time, pipe):
    """Write benchmark_info.json (arguments, hardware, timing, counters).
    Args:
        run_dir: Directory where the benchmark_info.json file will be saved.
        args: Namespace object containing command-line arguments.
        n_mpi: Number of MPI processes used in the run.
        status: Status of the run (e.g., "success" or "error").
        run_wall_time: Total wall time of the run in seconds.
        pipe: Instance of ClusterLikelihoodPipeline used in the run.
    """
    with open(os.path.join(run_dir, os.path.basename(args.priors_path))) as f:
        priors_content = f.read()
    info = {
        "timestamp_end": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "run_wall_time_s": run_wall_time,
        "arguments": vars(args),
        "command_line": sys.argv,
        "priors_file": os.path.basename(args.priors_path),
        "priors_content": priors_content,
        "mpi_processes": n_mpi,
        "loglike_calls_success": pipe.loglike_call_count,
        "loglike_calls_error": pipe.loglike_error_count,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_model": _cpu_model(),
            "cpu_count_logical": os.cpu_count(),
            "cpu_affinity": len(os.sched_getaffinity(0)),
            "total_memory_kb": _meminfo_total(),
            "load_average": os.getloadavg(),
            "env": {
                k: os.environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "SLURM_JOB_ID",
                    "SLURM_CPUS_PER_TASK",
                    "SLURM_NTASKS",
                )
            },
        },
        "packages": _package_versions(
            ["gpry", "numpy", "camb", "firecrown", "cosmosis", "mpi4py"]
        ),
    }
    with open(os.path.join(run_dir, "benchmark_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _meminfo_total():
    try:
        with open("/proc/meminfo") as f:
            return int(f.readline().split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _package_versions(names):
    from importlib import metadata

    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out
