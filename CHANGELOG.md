# Changelog of SyNCoPy
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
Bugfix release

### NEW
- Added experimental loading functionality for NWB 2.0 files
- Added experimental loading functionality for Matlab mat files
- Added support for "scalar" selections, i.e., things like `selectdata(trials=0)`
  or `data.selectdata(channels='mychannel')`
- Added command line argument "--full" for more granular testing: the new default
  for running the testing pipeline is to execute a trimmed-down testing suite that
  does not probe all possible input permutations but focuses on the core functionality
  without sacrificing coverage.

### CHANGED
- Renamed `_selection` class property to `selection`
- Made plotting routines matplotlib 3.5 compatible
- The output of `show` is now automatically squeezed (i.e., singleton dimensions
  are removed from the returned array).

### REMOVED
- Do not parse scalars using `numbers.Number`, use `numpy.number` instead to
  catch Boolean values
- Do not raise a `SPYTypeError` if an arithmetic operation is performed using
  objects of different numerical types (real/complex; closes #199)

### DEPRECATED
- Removed loading code for ESI binary format that is no longer supported
- Repaired top-level imports: renamed `connectivity` to `connectivityanalysis`
  and the "connectivity" module is now called "nwanalysis"
- Included `conda clean` in CD pipeline to avoid disk fillup by unused conda
  packages/cache
- Inverted `selectdata` messaging policy: only actual on-disk copy operations
  trigger a `SPYInfo` message (closes #197)
- Matched selector keywords and class attribute names, i.e., selecting channels
  is now done by using a `select` dictionary with key `'channel'` (not `'channels'`
  as before). See the documentation of `selectdata` for details.
- Retired travis CI tests since free test runs are exhausted. Migrated to GitHub
  actions (and re-included codecov)

### FIXED
- The `trialdefinition` arrays constructed by the `Selector` class were incorrect
  for `SpectralData` objects without time-axis, resulting in "empty" trials. This
  has been fixed (closes #207)
- Repaired `array_parser` to adequately complain about mixed-type arrays (closes #211)

## [v0.20] - 2022-01-18
Major Release

### NEW
- Added Connectivity submodule with `csd`, `granger` and `coh` measures
- Added new `CrossSpectralData` class for connectivity data
- Added Superlet spectral estimation method to `freqanalysis`
- Added arithmetic operator overloading for SyNCoPy objects: it is now possible
  to perform simple arithmetic operations directly, e.g.,``data1 + data2``.
- Added equality operator for SyNCoPy objects: two objects can be parsed for
  identical contents using the "==" operator
- Added full object padding functionality
- Added support for user-controlled in-place selections
- Added `show` class method for easy data access in all SyNCoPy objects
- Added de-trending suppport in `freqanalysis` via the `polyremoval` keyword
- New interface for synthetic data generation: using a list of NumPy arrays for
  instantiation interprets each array as `nChannels` x `nSamples` trial data
  which are combined to generate a `AnalogData` object
- Made SyNCoPy PEP 517 compliant: added pyproject.toml and modified setup.py
  accordingly
- Added IBM POWER testing pipeline (via dedicated GitLab Runner)

### CHANGED
- Multi-tapering now works with smoothing frequencies in Hz
- Streamlined padding interface

### REMOVED
- Retired tox in `slurmtest` CI pipeline in favor of a "simple" pytest testing
  session due to file-locking problems of tox environments on NFS mounts

### DEPRECATED
- Removed ACME from source repository: the submodule setup proved to be too
  unreliable and hard to maintain. ACME is now an optional (but recommended)
  dependency of SyNCoPy

### FIXED
- Non-standard `dimord` objects are now parsed and processed by `ComputationalRoutine`
- Impromptu padding performed by `freqanalysis` is done in a more robust way
- Stream-lined GitLab Runner setup: use cluster-wide conda instead of local
  installations (that differ slightly across runners) and leverage `tox-conda`
  to fetch pre-built dependencies

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
