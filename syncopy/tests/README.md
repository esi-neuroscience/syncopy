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

