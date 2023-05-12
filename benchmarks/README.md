# Syncopy Performance Benchmarks

This directory contains the Syncopy performance benchmarks, implemented with [Airspeed Velocity](https://asv.readthedocs.io), i.e., the `asv` Python package.

Note: The current version of `asv` does not seem to work with poetry at all, and it still defaults to using the rather outdated `setup.py` method instead of `pyproject.toml`. We do not hava a `setup.py`, nor do we want to ship one, so we convert our `pyproject.toml` to `setup.py` on the fly before running the performance benchmarks.

## Running the benchmarks

To run the benchmarks for the latest commit on your current branch:

```shell
cd repo/benchmarks/
pip install dephell
dephell convert deps --from-path ../pyproject.toml --from-format pyproject --to-path ../setup.py --to-format setuppy
pip install asv
asv run HEAD^!
```


## Common issues

If you are getting errors when running the benchmarks, e.g., `no module named syncopy`, you most likely have changed something with the `asv` configuration that broke the installation. In addition to fixing that, you will have to manually delete the old environments so that `asv` creates new ones afterwards:

```shell
rm -rf .asv/env
```



