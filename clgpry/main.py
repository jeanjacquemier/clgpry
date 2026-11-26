import argparse
import os
import shutil
from datetime import datetime

from .pipeline import ClusterLikelihoodPipeline
from .runner import exec_runner

from .benchmark import write_benchmark_file


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
        required=True,
    )
    parser.add_argument(
        "--data_sacc_path",
        type=str,
        help="Full path name for sacc input data",
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
    priors_abs = os.path.abspath(args.priors_path)
    likelihood_source_abs = os.path.abspath(args.likelihood_source)
    data_sacc_path_abs = os.path.abspath(args.data_sacc_path)
    os.environ["CLG_DATA"] = data_sacc_path_abs
    os.makedirs(run_dir, exist_ok=True)
    if is_main:
        shutil.copy2(priors_abs, run_dir)
        shutil.copy2(likelihood_source_abs, run_dir)
    os.chdir(run_dir)

    if is_main:
        print(f"Running GPry with {size} MPI process(es)")
        print(f"INFO: Output directory: {run_dir}")

    runner, pipe, status, wall_time = exec_runner(
        args.std_path, priors_abs, is_main, args.test_only
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


if __name__ == "__main__":
    main()


"""
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
"""
