import argparse
import json
import os
import platform
import shutil
import socket
import sys
import time
import numpy as np
from datetime import datetime


from .pipeline import ClusterLikelihoodPipeline


def write_benchmark_file(run_dir, args, n_mpi, status, run_wall_time, pipe):
    """Write benchmark_info.json (arguments, hardware, timing, counters)."""
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
            "env": {k: os.environ.get(k) for k in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_NTASKS")},
        },
        "packages": _package_versions(["gpry", "numpy", "camb", "firecrown", "cosmosis", "mpi4py"]),
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


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run the GPry Runner pipeline with configurable parameters."
    )
    parser.add_argument(
        "--priors-path",
        type=str,
        help="Files for prior files (default: data/priors_file.ini)",
        required=True,
    )
    parser.add_argument(
        "--std-path",
        type=str,
        help="Path to the CosmoSIS standard library",
        required=True        
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/intro",
        help="Directory for Runner checkpoints (default: output/intro)",
    )
    parser.add_argument(
        "--likelihood-source",
        type=str,
        default="likelihood_file.py",
        help="Path or source identifier for Firecrown likelihood configuration",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run a test of the likelihood evaluation without executing the full pipeline", 
    )

    args = parser.parse_args()


    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
    except ImportError:
        rank, size = 0, 1
    is_main = rank == 0

    # Every rank builds its own pipeline: each process evaluates the
    # likelihood for the points it is assigned, so each needs its own
    # Consistency/CAMB/Firecrown instances. Only rank 0 should be chatty.
    pipe = ClusterLikelihoodPipeline(
        likelihood_source=args.likelihood_source,
        std_path=args.std_path,
        priors_path=args.priors_path,
        verbose=False,
        camb_options=dict(verbose=is_main),
    )

    # Create a timestamped output folder (same name on all ranks) and run
    # from inside it, so that images/, loglike_histogram.png and
    # loglike_values.npy all end up there. Done after the pipeline is built
    # so relative input paths are resolved from the original directory.
    run_dir = datetime.now().strftime("%Y%m%d_%H%M%S") if is_main else None
    if size > 1:
        run_dir = MPI.COMM_WORLD.bcast(run_dir, root=0)
    run_dir = os.path.abspath(run_dir)
    priors_abs = os.path.abspath(args.priors_path)
    os.makedirs(run_dir, exist_ok=True)
    if is_main:
        shutil.copy2(priors_abs, run_dir)
    os.chdir(run_dir)

    if is_main:
        print(f"Running GPry with {size} MPI process(es)")
        print(f"INFO: Output directory: {run_dir}")

    # Runner detects MPI by itself: no explicit MPI code needed here.
    from gpry import Runner
    params_name = pipe.get_param_names(use_latex=True, plot_safe=True)
    print(f'INFO: {len(params_name)} parameters NOT fixed: {params_name}')
    print(f'INFO: Bounds: {pipe.bounds}')
    # runner = Runner(pipe.loglike, pipe.bounds, params=params_name)#, checkpoint=args.checkpoint, load_checkpoint="resume")
    status = "success"
    t_start = time.perf_counter()
    try:
        if not args.test_only:
            runner.run()
        else:
            for i in range(2):
                pipe.test(delta = np.random.rand()/1000)
    except Exception as e:
        status = f"error: {e!r}"
        print(f'GPry Runner error: {e!r}')
        pipe.com_abort()
    finally:
        wall_time = time.perf_counter() - t_start
        if is_main:
            pipe.print_loglike_counter()
            
            try:
                pipe.save_loglike_histogram()
            except Exception as e:
                print(f'Error save_loglike_histogram: {e!r}')
            
            values_array = np.array(pipe.loglike_values)
            with open('loglike_values.npy', 'wb') as f:
                np.save(f, values_array)

            write_benchmark_file(run_dir, args, size, status, wall_time, pipe)
    # Plot / save only from the main process to avoid N processes
    # writing the same files.
    if is_main and not args.test_only:
        runner.plot_progress()
        runner.plot_mc()


if __name__ == "__main__":
    main()


'''
runner = Runner(
    logLkl,
    bounds,
    options={
        # If there is multimodality, we are more likely to be
        # exploring simultaneously a couple of interesting areas
        # so we can evaluate more points per iteration
        "n_points_per_acq": "2d",
        # We will probably need more training samples than default
        # to represent a complicated posterior surface
        "max_total": 400},
    surrogate={
        "regressor": {
            # We want to adapt hyperparameters more often, to make
            # it more likely to converge earlier
            "n_restarts_optimizer": 20}},
    gp_acquisition={
        "NORA": {
            # We want to make the acquisition function more
            # exploratory (higher zeta_scaling)
            "acq_func": {"LogExp": {"zeta_scaling": 1.1}},
            # We want to re-run the nested sampler exploration as
            # often as possible, because we expect frequent changes
            "mc_every": 1}},
    convergence_criterion={
        # We want to make the CorrectCounter criterion necessary,
        # not just sufficient, so that it detects when a new region
        # is being explored. But we can also relax it a bit.
        "CorrectCounter": {"policy": "n", "reltol": 0.05, "abstol": "0.05s"},
        "GaussianKL": {"policy": "n"},
        "TrainAlignment": {"policy": "n"},},
    mc={
        # In the final MC run, we want more live points for a
        # better exploration of all the modes
        "nested": {"nlive": "100d"}},
    # Just for diagnostics. It will severely slow down the run if uncommented
    # plots={
    #     # Let's do a corner plot per iteration
    #     "corner": True, "timing": False, "convergence": False,
    #     "trace": False, "slices": False, "ext": "png"},
    checkpoint=checkpoint,
    load_checkpoint="overwrite")
'''
