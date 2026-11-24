# clgpry

`clgpry` is a research-stage Python package for evaluating galaxy-cluster
cosmology likelihoods. It connects CosmoSIS consistency and CAMB modules to a
Firecrown likelihood, with parameter bounds and defaults loaded from an INI
file. The pipeline is designed to provide a likelihood function for Bayesian
sampling workflows such as those built with GPRy.

> **Status:** Alpha. A working installation also needs compatible external
> CosmoSIS, Firecrown, CROW, and cluster-data resources. These are not all
> included in this repository.

## Requirements

- Python 3.8 or later
- The Python dependencies listed in `pyproject.toml` (including NumPy,
  Matplotlib, CosmoSIS, Firecrown, and GPRy)
- A CosmoSIS standard library containing the Consistency and CAMB modules
- Compatible Firecrown/CROW dependencies, including the cluster modeling
  libraries used by the likelihood
- A Firecrown likelihood source and the required cluster SACC data file

The precise installation steps for the scientific dependencies depend on your
platform and the versions of CosmoSIS, Firecrown, and CROW you use. Set up that
environment first, then install this package:

```bash
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

## Configuration and data

The default cosmological and Firecrown parameters and their sampling bounds
are read from [`data/priors_file.ini`](data/priors_file.ini). In each section,
a single numeric value fixes a parameter; three values (`minimum central
maximum`) mark it as sampled and define its bounds. The pipeline preserves
the order of sampled parameters as they appear in the INI sections.

[`data/sampler_file.ini`](data/sampler_file.ini) is an example CosmoSIS sampler
configuration. Review its paths and options for your environment before using
it: it contains machine-specific paths.

The repository does not include the cluster SACC dataset or a likelihood
source. In particular, `clgpry.likelihood.build_likelihood` currently loads
`./data/clusters_sacc_file_cov.sacc` relative to the process working directory.
Provide that file in the expected location (or adapt the likelihood source for
your data) before building or running that likelihood.

## Using the pipeline
### From the command line
```bash
clgpry-run -h
```

### Inside a python implementation.
standard library. Supply a parameter vector in the order returned by
`get_param_names()`:

```python
from clgpry import ClusterLikelihoodPipeline

pipeline = ClusterLikelihoodPipeline(
    likelihood_source="/path/to/your/likelihood_source.py",
    std_path="/path/to/cosmosis-standard-library",
)

print(pipeline.get_param_names())
print(pipeline.bounds)

# Values must follow get_param_names() and lie strictly inside pipeline.bounds.
params = [0.22, 0.8, 0.958236982, 0.04552506]
log_likelihood = pipeline.loglike(params)
print(log_likelihood)
```

The example vector corresponds to the sampled parameters in the included
priors file. Update it if you change those priors. Values exactly on a bound
are invalid. A `ValueError` during evaluation is recorded and returned as
negative infinity; other pipeline failures are raised.

The constructor also accepts `camb_options`, `likelihood_options`, and
`verbose`. For example, likelihood options can toggle the cluster observables
used by the Firecrown likelihood:

```python
pipeline = ClusterLikelihoodPipeline(
    likelihood_source="/path/to/your/likelihood_source.py",
    std_path="/path/to/cosmosis-standard-library",
    likelihood_options={
        "use_cluster_counts": True,
        "use_mean_log_mass": False,
        "use_mean_deltasigma": True,
        "use_mean_reduced_shear": False,
    },
)
```

For a parameter vector, `block_from_params(params)` creates a CosmoSIS
`DataBlock`, and `run(block)` executes the Consistency, CAMB, and Firecrown
stages. A fresh block is required for each run. The pipeline also provides
`print_loglike_counter()` and `save_loglike_histogram()` for inspecting
likelihood evaluations.

## Development

Install the development extras and run the test suite with:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## License

The project metadata declares the MIT License.
