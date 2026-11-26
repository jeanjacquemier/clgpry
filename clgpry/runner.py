from clgpry.pipeline import ClusterLikelihoodPipeline
import argparse
import numpy as np
import time

from gpry import Runner

from .plot import save_loglike_histogram


def exec_runner(
    std_path: str,
    priors_path: str,
    is_main: bool,
    test_only: bool,
):
    """Execute the GPry runner with the likelihood and bounds from the pipeline.
    Args:
        std_path: Path to the CosmoSIS standard library.
        priors_path: Path to the prior file.
        is_main: Boolean indicating if this is the main MPI rank.
        test_only: Boolean indicating if only a test run should be performed.
    Returns:
        runner: Instance of the GPry Runner.
        pipe: Instance of the ClusterLikelihoodPipeline.
        status: Status of the run (e.g., "success" or "error").
        wall_time: Wall clock time taken for the run.
    """
    # Runner detects MPI by itself: no explicit MPI code needed here.

    # Every rank builds its own pipeline: each process evaluates the
    # likelihood for the points it is assigned, so each needs its own
    # Consistency/CAMB/Firecrown instances. Only rank 0 should be chatty.
    pipe = ClusterLikelihoodPipeline(
        std_path=std_path,
        priors_path=priors_path,
        verbose=False,
        camb_options=dict(verbose=is_main),
    )

    params_name = pipe.get_param_names(use_latex=False, plot_safe=True)
    print(f"INFO: {len(params_name)} parameters NOT fixed: {params_name}")
    print(f"INFO: Bounds: {pipe.bounds}")
    runner = Runner(
        pipe.loglike, pipe.bounds, params=params_name
    )  # , checkpoint=args.checkpoint, load_checkpoint="resume")
    status = "success"
    t_start = time.perf_counter()
    try:
        if not test_only:
            runner.run()
        else:
            for i in range(2):
                pipe.test(delta=np.random.rand() / 1000)
    except Exception as e:
        status = f"error: {e!r}"
        print(f"GPry Runner error: {e!r}")
        pipe.com_abort()
    finally:
        wall_time = time.perf_counter() - t_start
        if is_main:
            values_array = np.array(pipe.loglike_values)
            pipe.print_loglike_counter()            
            with open("loglike_values.npy", "wb") as f:
                np.save(f, values_array)

    return runner, pipe, status, wall_time
