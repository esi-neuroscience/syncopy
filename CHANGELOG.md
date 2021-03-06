# Changelog of SyNCoPy
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [v0.1b2] - 2020-01-15
Housekeeping and maintenance release

### NEW
- Included ACME as SyNCoPy submodule: all ESI-HPC cluster specific code has
  been migrated to the new ACME package, see https://github.com/esi-neuroscience/acme
- Better late than never: added this CHANGELOG file

### CHANGED
- Modified GitLab CI Pipeline Setup + version handling: use `setuptools_scm`
  to populate `spy.__version__` instead of hard-coding a version string
  in the package `__init__.py`; this makes test-uploads to PyPI-Test infinitely
  easier since `setuptools_scm` takes care of generating non-conflicting
  package versions.
- Modified packaging setup and adapted modular layout to account for new
  submodule ACME

### REMOVED
- Deleted ESI-specific `dask_helpers.py` module (migrated to ACME)

### DEPRECATED
- Cleaned up dependencies: removed all `jupyter`-packages from depencency
  list to not cause (unnecessary) conflicts in existing Python environments

### FIXED
- Repaired CI pipelines
- Repaired h5py version mismatch: pin SyNCoPy to `hypy` versions greater than
  2.9 but less than 3.x
- Pin SyNCoPy to Python 3.8.x (Python 3.9 currently triggers too many dependency
  conflicts)

## [v0.1b1] - 2020-10-23
First public pre-release of SyNCoPy on PyPI and GitHub.

### NEW
- Included `selectdata` as a `computeFunction` that uses the parallelization
  framework in `ComputationalRoutine` to perform arbitrary data-selection tasks
  (including but not limited to unordered lists, repetitions and removals).
- Included time-frequency analysis routines `mtmconvol` and `wavelet`
- Added plotting functionality: functions `singlepanelplot` and `multiplanelplot`
  allow quick visual inspection of `AnalogData` and `SpectralData` objects
- Added support to process multiple SyNCoPy objects in a single meta-function
  call (all decorators have been modified accordingly)
- Introduced standardized warning messages via new class `SPYWarning`
- Included (more or less) extensive developer docs
- Added Travis CI and included badges on GitHub landing page
- New convenience scripts to ease developing/testing
- New conda.yml file + script for consolidating conda/pip requirements: all
  of SyNCoPy's dependencies are now collected in `syncopy.yml`, the respective
  pip-specific requirements.txt and requirements-test.txt files are generated
  on the fly by a new function `conda2pip` that relies on ruamel.yaml (new
  required dependency for building SyNCoPy)
- New GitLab CI directive for uploading SyNCoPy to PyPI
- Included GitHub templates for new issues/pull requests
- SyNCoPy docu is now hosted on readthedocs (re-directed from syncopy.org)
- New logo + icon

### CHANGED
- Made `cluster_cleanup` more robust (works with `LocalCluster` objects now)
- Made data-parser more feature-rich: check for emptiness, parse non-data
  datasets etc.
- Made `generate_artificial_data` more robust: change usage of random number
  seed to allow persistent comparisons across testing runs
- Updated CI dependencies (SyNCoPy now requires NumPy 1.18 and Matplotlib 3.3.x)

### REMOVED
- All *.py-file headers have been removed
- Removed examples sub-module from main package (examples will be part of a
  separate repo)

### DEPRECATED
- Wiped all hand-crafted array-matching routines; use `best_match` instead
- Do not use `pbr` in the build system any more; rely instead on up-to-date
  setuptools functionality
- Retired memory map support and raw binary data reading routines

### FIXED
- Improved temporary storage handling so that dask workers that import the
  package do not repeat all temp-dir I/O tests (and potentially run into
  dead-locks or race conditions)

## [v0.1a1] - 2019-10-14
Preview alpha release of SyNCoPy for first ESI-internal tryout workshop.

### NEW
- Added routines `esi_cluster_setup` and `cluster_cleanup` to facilitate
  using SLURM from within SyNCoPy
- Included new `FauxTrial` class and `_preview_trial` class methods to
  permit quick and performant compute dry-runs
- Included a `select` keyword to allow for in-place selections that are applied
  on the fly in any meta-function via a new decorator. The heavy lifting is performed
  by a new `Selector` class
- Re-worked the `specest` package: `mtmfft` is now fully functional
- Overhauled HTML documentation

### CHANGED
- New layout of SyNCoPy objects on disk: introduction of Spy-containers supporting
  multiple datasets/objects within the same folder
- First working implementation of `spy.load` and `spy.save`
- Use dask bags instead of arrays in parallelization engine to permit more
  flexible distribution of data across workers
- Re-worked `trialdefinition` mechanics: attach the full `trialdefinition` array
  to objects and fetch relevant information on the fly: `BaseData._trialdefinition`
  unifies `sampleinfo`, `t0` and `trialinfo` and calls `definetrial`

### REMOVED
- Removed `dimlabels` property

### DEPRECATED
- Retired Dask arrays in `ComputationalRoutine`; use dask bags instead

### FIXED
- Flipped sign of offsets in `trialdefinition` to be compatible w/FieldTrip
- Enforced PEP8 compliance
- Cleaned up constructor of `BaseData` to prohibit accessing uninitialized attributes

## [v0.1a0] - 2019-07-20
Internal pre-alpha release of SyNCoPy. Prototypes of data format, user-interface
and parallelization framework are in place.

### NEW
- Class structure is laid out, meta-functions are present but mostly place-holders
- Support FieldTrip-style calling syntax via `cfg` "structures" (the keys of which are
  "unwrapped" by a corresponding decorator)
- Preliminary I/O capabilities implemented, objects can be written/read
  to/from HDF5
- First prototype of parallelization framework based on Dask
- Custom traceback that is enabeld as soon as SyNCoPy is imported: do not
  spill hundreds of lines to STDOUT, instead highlight most probable cause
  of error and explain how to get to full traceback if wanted
- Basic session management to ensure concurrent SyNCoPy sessions only access
  their own data
