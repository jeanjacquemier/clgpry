import os

# One MPI process = one core. Without this, CAMB/numpy (OpenMP/BLAS) spawn
# extra threads in every rank and oversubscribe the node.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import sys
import importlib.util
import configparser

import numpy as np


from cosmosis.datablock import DataBlock, option_section
from firecrown.connector.cosmosis.likelihood import FirecrownLikelihood


def _load_priors_from_ini(ini_path, verbose=False):
    """Load cosmological and firecrown parameters from priors INI file.
    
    Args:
        ini_path: Path to the priors INI file.
    
    Returns:
        Tuple of (default_cosmo_dict, default_fc_dict, bounds_list, param_names_list)
    """
    config = configparser.ConfigParser()
    config.read(ini_path)

    default_cosmo = {}
    default_fc = {}
    param_names = []
    bounds = []

    
    # Parse cosmological_paramete   rs section
    if "cosmological_parameters" in config:
        for key, value in config["cosmological_parameters"].items():
            parts = value.split()
            if len(parts) == 1:
                # Fixed parameter
                default_cosmo[key] = float(parts[0])
            elif len(parts) == 3:
                # Sampled parameter: min central max
                default_cosmo[key] = float(parts[1])
                param_names.append(("cosmological_parameters", key))
                bounds.append([float(parts[0]), float(parts[2])])
    
    # Parse firecrown_number_counts section
    if "firecrown_number_counts" in config:
        for key, value in config["firecrown_number_counts"].items():
            parts = value.split()
            if len(parts) == 1:
                # Fixed parameter
                default_fc[key] = float(parts[0])
            elif len(parts) == 3:
                # Sampled parameter: min central max
                default_fc[key] = float(parts[1])
                param_names.append(("firecrown_number_counts", key))
                bounds.append([float(parts[0]), float(parts[2])])
    
    assert len(bounds) > 1, "At least too parameters need to be NOT fixed."

    if verbose == True:
        print('\n========== Default Cosmological parameters values ===========')
        for key, value in default_cosmo.items():
            print(key, value)    
        print('\n========== Default Firecrown number counts parameters values ===========')
        for key, value in default_fc.items():
            print(key, value)    
        print('\n========== GPRy Bounds ===========')
        print(f'There are {len(bounds)} bounds defined:')
        for param, bound in zip(param_names, bounds):
            print(param, bound)

        
        
    
    return default_cosmo, default_fc, bounds, param_names


class ClusterLikelihoodPipeline:
    """Run Consistency, CAMB, and Firecrown as a sampled likelihood.

    The three modules are initialized once. A fresh DataBlock is created for
    each evaluation because DataBlocks are write-once and cannot be reused.

    Args:
        std_path: Path to the CosmoSIS standard library.
        camb_options: Optional CAMB configuration values overriding defaults.
        likelihood_options: Optional likelihood configuration values
            overriding defaults.
        verbose: Whether to print stage and likelihood progress.
    """

    CP = "cosmological_parameters"
    FNC = "firecrown_number_counts"


    # LaTeX labels for parameter names
    LATEX_LABELS = {
        "omega_c": r"$\Omega_c$",
        "sigma_8": r"$\sigma_8$",
        "omega_b": r"$\Omega_b$",
        "h0": r"$h_0$",
        "n_s": r"$n_s$",
        "tau": r"$\tau$",
        "w": r"$w$",
        "wa": r"$w_a$",
        "omega_k": r"$\Omega_k$",
        "mass_distribution_mu0": r"$\mu_0$",
        "mass_distribution_mu1": r"$\mu_1$",
        "mass_distribution_mu2": r"$\mu_2$",
        "mass_distribution_sigma0": r"$\sigma_0$",
        "mass_distribution_sigma1": r"$\sigma_1$",
        "mass_distribution_sigma2": r"$\sigma_2$",
        "cluster_theory_cluster_concentration": "Concentration",
        "completeness_a_n": r"$a_n$",
        "completeness_b_n": r"$b_n$",
        "completeness_a_logm_piv": r"$a_{logM}^{piv}$",
        "completeness_b_logm_piv": r"$b_{logM}^{piv}$",
    }

    # Plot-safe labels (compatible with GPRy plot_mc method)
    PLOT_LABELS = {
        "omega_c": r"\Omega_c",
        "sigma_8": r"\sigma_8",
        "omega_b": r"\Omega_b",
        "h0": r"h_0",
        "n_s": r"n_s",
        "tau": r"\tau",
        "w": r"w",
        "wa": r"w_a",
        "omega_k": r"\Omega_k",
        "mass_distribution_mu0": r"\mu_0",
        "mass_distribution_mu1": r"\mu_1",
        "mass_distribution_mu2": r"\mu_2",
        "mass_distribution_sigma0": r"\sigma_0",
        "mass_distribution_sigma1": r"\sigma_1",
        "mass_distribution_sigma2": r"\sigma_2",
        "cluster_theory_cluster_concentration": "Concentration",
        "completeness_a_n": r"a_n",
        "completeness_b_n": r"b_n",
        "completeness_a_logm_piv": r"a_{logM}^{piv}",
        "completeness_b_logm_piv": r"b_{logM}^{piv}",
    }
   

    def __init__(self,
                 std_path,
                 priors_path,
                 camb_options=None, likelihood_options=None,
                 verbose=False):
        """Initialize the pipeline modules and likelihood configuration.

        Args:
            std_path: Path to the CosmoSIS standard library.
            camb_options: Optional CAMB configuration overrides.
            likelihood_options: Optional Firecrown likelihood configuration
                overrides.
            verbose: Whether to print progress during evaluations.
        """
        self.std_path = os.path.expanduser(std_path)
        self.verbose = verbose

        # Load parameters from priors file
        self.DEFAULT_COSMO, self.DEFAULT_FC, self.BOUNDS, self.PARAM_NAMES = _load_priors_from_ini(priors_path)

        self.bounds = [list(b) for b in self.BOUNDS]
        self.loglike_call_count = 0
        self.loglike_error_count = 0
        self.loglike_values = []



        self._setup_consistency()
        self._setup_camb(camb_options)
        self._setup_likelihood(likelihood_options)

    
    def get_param_names(self, use_latex=False, plot_safe=False):
        """Extract parameter names from PARAM_NAMES tuples.
        
        Args:
            use_latex: If True, replace parameter names with LaTeX labels
                (with $ delimiters for matplotlib).
            plot_safe: If True, replace parameter names with plot-safe labels
                (LaTeX without $ delimiters, compatible with GPRy plot_mc).
                Takes precedence over use_latex if both are True.
        
        Returns:
            List of parameter names (second element of each PARAM_NAMES tuple).
            If use_latex=True, names are replaced with LaTeX labels formatted
            for matplotlib: ['$\\Omega_c$', '$\\sigma_8$', ...]
            If plot_safe=True, names are replaced with plot-safe labels
            compatible with GPRy: ['\\Omega_c', '\\sigma_8', ...]
            Otherwise, returns plain parameter names.
            
        Example:
            >>> pipe = ClusterLikelihoodPipeline(...)
            
            # Plain parameter names
            >>> names = pipe.get_param_names()
            >>> names
            ['omega_c', 'sigma_8', 'mass_distribution_mu0', ...]
            
            # LaTeX labels for matplotlib
            >>> latex_names = pipe.get_param_names(use_latex=True)
            >>> latex_names
            ['$\\Omega_c$', '$\\sigma_8$', '$\\mu_0$', ...]
            
            # Plot-safe labels for GPRy plot_mc
            >>> plot_names = pipe.get_param_names(plot_safe=True)
            >>> plot_names
            ['\\Omega_c', '\\sigma_8', '\\mu_0', ...]
        """
        param_names = [name for section, name in self.PARAM_NAMES]
        
        if plot_safe:
            param_names = [
                ClusterLikelihoodPipeline.PLOT_LABELS.get(name, name) for name in param_names
            ]
        elif use_latex:
            param_names = [
                ClusterLikelihoodPipeline.LATEX_LABELS.get(name, name) for name in param_names
            ]
        return param_names

    # ------------------------------------------------------------------ setup
    def _load_module(self, relpath, name):
        """Load a standard-library module interface from a file path.

        Args:
            relpath: Module path relative to the standard-library directory.
            name: Name to assign to the loaded module.

        Returns:
            The loaded Python module.
        """
        path = os.path.join(self.std_path, relpath)
        sys.path.insert(0, os.path.dirname(path))   # for sibling imports
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _setup_consistency(self):
        """Load and configure the CosmoSIS consistency module."""
        self.consistency_mod = self._load_module(
            "utility/consistency/consistency_interface.py",
            "consistency_interface")
        opts = DataBlock()
        opts[option_section, "verbose"] = self.verbose
        self.cons_cfg = self.consistency_mod.setup(opts)

    def _setup_camb(self, camb_options=None):
        """Load and configure CAMB, applying any supplied option overrides.

        Args:
            camb_options: Optional CAMB configuration values overriding the
                pipeline defaults.
        """
        self.camb_mod = self._load_module(
            "boltzmann/camb/camb_interface.py", "camb_interface")
        defaults = dict(mode="all", lmax=2500, feedback=0,
                        zmin=0.0, zmax=10.0, nz=100,
                        kmin=1e-4, kmax=10.0, nk=1000, verbose=True)
        defaults.update(camb_options or {})
        opts = DataBlock()
        for k, v in defaults.items():
            opts[option_section, k] = v
        self.camb_cfg = self.camb_mod.setup(opts)

    def _setup_likelihood(self, likelihood_options=None):
        """Configure the Firecrown likelihood.

        Args:
            likelihood_options: Optional likelihood configuration values
                overriding the defaults.
        """
        defaults = dict(
            likelihood_source="likelihood_file.py",
            sampling_parameters_sections=self.FNC,
            # read by build_likelihood via build_parameters.get_bool
            use_cluster_counts=True,
            use_mean_log_mass=False,
            use_mean_deltasigma=True,
            use_mean_reduced_shear=False,
        )
        defaults.update(likelihood_options or {})
        opts = DataBlock()
        for k, v in defaults.items():
            opts[option_section, k] = v
        self.firecrown_likelihood = FirecrownLikelihood(opts)

    # --------------------------------------------------------------- one eval
    def make_block(self, cosmo=None, fc=None):
        """Create a fresh DataBlock populated with default and override values.

        Args:
            cosmo: Optional cosmological parameter overrides.
            fc: Optional Firecrown parameter overrides.

        Returns:
            A new DataBlock ready for execution by the pipeline.
        """
        block = DataBlock()
        for k, v in {**self.DEFAULT_COSMO, **(cosmo or {})}.items():
            block[self.CP, k] = v
        for k, v in {**self.DEFAULT_FC, **(fc or {})}.items():
            block[self.FNC, k] = v
        return block

    def block_from_params(self, params):
        """Convert a sampled parameter vector into a ready-to-run DataBlock.

        Args:
            params: Values ordered according to ``PARAM_NAMES``.

        Returns:
            A new DataBlock containing the sampled values and defaults.

        Raises:
            ValueError: If the vector length does not match ``PARAM_NAMES``.
        """
        if len(params) != len(self.PARAM_NAMES):
            raise ValueError(f"expected {len(self.PARAM_NAMES)} parameters, "
                             f"got {len(params)}")
        cosmo, fc = {}, {}
        for (section, name), value in zip(self.PARAM_NAMES, params):
            (cosmo if section == self.CP else fc)[name] = float(value)
        return self.make_block(cosmo=cosmo, fc=fc)

    def check_bounds(self, params):
        """Check each parameter value against its corresponding strict bounds.

        Args:
            params: Values ordered according to ``PARAM_NAMES``.

        Raises:
            ValueError: If a checked value is outside or exactly on its bounds.
        """
        for (section, name), p, (lo, hi) in zip(self.PARAM_NAMES, params,
                                                self.bounds):
            if not (lo < p < hi):
                raise ValueError(f"{name}={p} outside ({lo}, {hi})")

    def run(self, block):
        """Execute all pipeline stages on a fresh DataBlock.

        Args:
            block: DataBlock to process. It must not have been used previously.

        Returns:
            The Firecrown log-likelihood stored in the DataBlock.

        Raises:
            RuntimeError: If any pipeline stage reports failure.
        """
        stages = [
            ("Consistency", lambda: self.consistency_mod.execute(block, self.cons_cfg)),
            ("CAMB",        lambda: self.camb_mod.execute(block, self.camb_cfg)),
            ("Likelihood",  lambda: self.firecrown_likelihood.execute(block)),
        ]
        for name, stage in stages:
            if self.verbose:
                print(f"-------< Stage {name} >--------------")
            if stage() != 0:
                raise RuntimeError(f"Stage {name} failed")
        return block["likelihoods", "firecrown_like"]

    # ------------------------------------------------------------- public API
    def loglike(self, params):
        """Evaluate the log-likelihood for a sampled parameter vector.

        Args:
            params: Values ordered according to ``PARAM_NAMES``.

        Returns:
            The log-likelihood, or negative infinity if evaluation raises
            ``ValueError``.
        """
        try:
            value = self.run(self.block_from_params(params))
            self.loglike_call_count += 1
        except ValueError:
            self.loglike_error_count += 1
            self.loglike_values.append(-np.inf)
            return -np.inf
        if self.verbose:
            print("=====> Likelihood result = ", value)
        self.loglike_values.append(value)
        return value

    def com_abort(self):
        """Abort the current MPI process.
        """
        self.comm.Abort(1)

    def print_loglike_counter(self):
        """Display the counter of loglike evaluations and errors.
        
        This method prints:
        - Number of successful loglike evaluations
        - Number of times loglike returned -inf due to ValueError
        - Total number of calls
        
        Returns:
            tuple: (successful_calls, error_calls, total_calls)
        """
        total_calls = self.loglike_call_count + self.loglike_error_count
        print("\n" + "="*60)
        print("Pipeline Execution Summary:")
        print(f"  Total successful loglike calls:        {self.loglike_call_count}")
        print(f"  Loglike calls with ValueError (-inf):  {self.loglike_error_count}")
        print(f"  Total loglike calls:                   {total_calls}")
        print("="*60 + "\n")
        return (self.loglike_call_count, self.loglike_error_count, total_calls)

    
    def test(self, delta=0.0):
        """Evaluate the likelihood near the reference parameter values.

        Args:
            delta: Value added to each reference parameter before evaluation.

        Returns:
            The log-likelihood returned by :meth:`loglike`.
        """
        params = np.array([0.22, 0.8]) + delta
        #params = np.array([0.22, 0.8, 3.33, 0.95, -0.023, 0.56, 0.0459, 0.0397]) + delta
        return self.loglike(params)


