## Syncopy Testing Routines

Frontends and general architecture, for explicit backend methods see `/backend` subdirectory.

### Run all

Just launch the `run_tests.sh` script.

### Manually start specific tests

Assuming you are in this `/test` directory,
amend your Python path with the `/syncopy` module directory:

```bash
export PYTHONPATH=../../
```

To run all connectivity tests except the parallel routines:

```bash
pytest -v test_connectivity.py -k 'not parallel'
```

### Running tests interactively in ipython

To run the tests interactively, first make sure you are in a proper environment to run syncopy (e.g., your conda syncopy-dev environment.)

Then start ipython from the Syncopy repo root, run a test file, and execute a test. E.g.:


```bash
cd ~/develop/syncopy
ipython
```

And in iypthon:

```python
run syncopy/tests/test_basedata.py # Just runs file, executes not tests.
TestBaseData().test_data_alloc() # Run a single test.
```



