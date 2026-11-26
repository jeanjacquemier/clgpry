from clgpry.pipeline import ClusterLikelihoodPipeline
import numpy as np
import time
import configparser

from gpry import Runner

from .plot import save_loglike_histogram

"""
  options={
        "n_initial": 30,        # initial samples drawn before the active-learning loop starts
        "max_initial": 1000,     # cap on the initial-sampling stage if the points are not "finite" enough
    },

n_initial : Nn_initial : Number of finite initial truth evaluations before starting the BO loop (default: 3 * number of dimensions)
max_initial : Maximum number of truth evaluations at initialization. If it is reached before n_initial finite points have been found, the run will fail. To avoid that, try decreasing the volume of your prior (default: 30 * (number of dimensions)**1.5).
max_total : Maximum number of attempted sampling points before the run stops. This is useful if you e.g. want to restrict the maximum computation resources (default: 70 * (number of dimensions)**1.5 or max_initial, whichever is largest).
max_finite : Maximum number of sampling points accepted into the GP training set before the run stops. This might be useful if you use the DontConverge convergence criterion, specifying exactly how many points you want to have in your GP. If you set this limit by hand and find that it is easily saturated, try decreasing the volume of your prior (default: max_total).
n_points_per_acq : Number of points which are aquired with Kriging believer for every acquisition step. It will be adjusted within a 20% tolerance to match a multiple of the number of parallel processes (default: number of dimensions).
fit_full_every : Number of iterations between full GP hyperparameters fits, including several restarts of the optimiser. Pass ‘np.inf’ or a large number to never refit with restarts (default : 2 * sqrt(number of dimensions)).
fit_simple_every : Similar to fit_full_every, but with a single optimiser run from the last optimum hyperparameters. Overridden by fit_full_every where it matches its periodicity. Pass np.inf or a large number to never refit from last optimum (default : 1, i.e. every iteration).e points have been found, the run will fail. To avoid that, try decreasing the volume of your prior (default: 30 * (number of dimensions)**1.5).
max_total : Maximum number of attempted sampling points before the run stops. This is useful if you e.g. want to restrict the maximum computation resources (default: 70 * (number of dimensions)**1.5 or max_initial, whichever is largest).
max_finite : Maximum number of sampling points accepted into the GP training set before the run stops. This might be useful if you use the DontConverge convergence criterion, specifying exactly how many points you want to have in your GP. If you set this limit by hand and find that it is easily saturated, try decreasing the volume of your prior (default: max_total).
n_points_per_acq : Number of points which are aquired with Kriging believer for every acquisition step. It will be adjusted within a 20% tolerance to match a multiple of the number of parallel processes (default: number of dimensions).
fit_full_every : Number of iterations between full GP hyperparameters fits, including several restarts of the optimiser. Pass ‘np.inf’ or a large number to never refit with restarts (default : 2 * sqrt(number of dimensions)).
fit_simple_every : Similar to fit_full_every, but with a single optimiser run from the last optimum hyperparameters. Overridden by fit_full_every where it matches its periodicity. Pass np.inf or a large number to never refit from last optimum (default : 1, i.e. every iteration).umber of finite initial truth evaluations before starting the BO loop (default: 3 * number of dimensions)

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


def _load_sampler_option_form_ini(sampler_ini_path: str) -> dict:
    """Load sampler options from INI file, specifically n_initial and max_initial.
    
    Reads the [grpy] section from the sampler INI configuration file and
    extracts the n_initial and max_initial parameters.
    
    Args:
        sampler_ini_path: Path to the sampler configuration INI file.
        
    Returns:
        dict: Dictionary containing 'n_initial' and 'max_initial' as integers.
              If either key is missing from the INI file, it will not be
              included in the returned dictionary.
              
    Raises:
        FileNotFoundError: If the sampler INI file does not exist.
        configparser.Error: If there's an error reading the INI file.
    """
    config = configparser.ConfigParser()
    config.read(sampler_ini_path)
    
    options = {}
    
    if config.has_section("grpy"):
        if config.has_option("grpy", "n_initial"):
            # Strip trailing commas and whitespace that might be in the INI file
            n_initial_str = config.get("grpy", "n_initial").strip().rstrip(",")
            options["n_initial"] = int(n_initial_str)
            
        if config.has_option("grpy", "max_initial"):
            # Strip trailing commas and whitespace that might be in the INI file
            max_initial_str = config.get("grpy", "max_initial").strip().rstrip(",")
            options["max_initial"] = int(max_initial_str)
    
    return options


def exec_runner(
    std_path: str,
    priors_path: str,
    sampler_ini: str,
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

    # Load sampler options from the INI file
    options = _load_sampler_option_form_ini(sampler_ini)
    print(f"INFO: Loaded sampler options: {options}")
    print(f"INFO: Bounds: {pipe.bounds}")
    runner = Runner(
        pipe.loglike, pipe.bounds, params=params_name,
        options=options, 
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
