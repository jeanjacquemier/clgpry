import argparse
import os
import shutil
import sys
import traceback
from datetime import datetime

from .pipeline import ClusterLikelihoodPipeline
from .runner import exec_runner

from .benchmark import write_benchmark_file


def _run():
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
        required=True,
    )
    parser.add_argument(
        "--data_sacc_path",
        type=str,
        help="Full path name for sacc input data",
        required=True,
    )
    parser.add_argument(
        "--sampler-ini",
        type=str,
        help="Path or source identifier for sampler.ini configuration",
        required=True,   
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

    # Create a timestamped output folder (same name on all ranks) and run
    # from inside it, so that images/, loglike_histogram.png and
    # loglike_values.npy all end up there. Done after the pipeline is built
    # so relative input paths are resolved from the original directory.
    run_dir = datetime.now().strftime("%Y%m%d_%H%M%S") if is_main else None
    if size > 1:
        run_dir = MPI.COMM_WORLD.bcast(run_dir, root=0)
    run_dir = os.path.abspath(run_dir)
    sampler_source_abs = os.path.abspath(args.sampler_ini)
    priors_abs = os.path.abspath(args.priors_path)
    likelihood_source_abs = os.path.abspath(args.likelihood_source)
    data_sacc_path_abs = os.path.abspath(args.data_sacc_path)
    os.environ["CLG_DATA"] = data_sacc_path_abs
    os.makedirs(run_dir, exist_ok=True)
    if is_main:
        shutil.copy2(sampler_source_abs, run_dir)
        shutil.copy2(priors_abs, run_dir)
        shutil.copy2(likelihood_source_abs, run_dir)
        
    os.chdir(run_dir)

    if is_main:
        print(f"Running GPry with {size} MPI process(es)")
        print(f"INFO: Output directory: {run_dir}")

    runner, pipe, status, wall_time = exec_runner(
        args.std_path, priors_abs,
        sampler_source_abs, is_main, args.test_only
    )
    if is_main:
        write_benchmark_file(run_dir, args, size, status, wall_time, pipe)
        # Plot / save only from the main process to avoid N processes
        # writing the same files.
        if not args.test_only:
            try:
                runner.plot_progress()
            except Exception as e:
                print(f"runner.plot_progress error: {e!r}")

            try: 
                runner.generate_mc_sample()
            except Exception as e:
                print(f"runner.generate_mc_sample error: {e!r}")
            try:
                runner.plot_mc()
            except Exception as e:
                print(f"runner.plot_mc error: {e!r}")

    # All ranks leave together; mpi4py finalizes MPI at interpreter exit.
    print(f'DEBUG mpi size={size}' )
    if size > 1:
        print('MPI.COMM_WORLD.Barrier()')
        MPI.COMM_WORLD.Barrier()


def main():
    """Run the pipeline; on any failure on any rank, abort the whole MPI job."""
    try:
        _run()
    except SystemExit as e:
        # argparse errors / explicit exits: only abort if non-zero
        if e.code not in (0, None):
            _abort_all()
        raise
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        _abort_all()
        raise


def _abort_all():
    try:
        from mpi4py import MPI
    except ImportError:
        return
    if MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.Abort(1)


if __name__ == "__main__":
    main()

