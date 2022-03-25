#!/bin/bash
# Some quick shortcuts to ease running Syncopy's testing pipeline

# First and foremost, check if `srun` is available
_useSLURM=$(command -v srun)

# Stuff only relevant in here
_self=$(basename "$BASH_SOURCE")
_selfie="${_self%.*}"
_ppname="<$_selfie>"

# Brief help message explaining script usage
usage()
{
    echo "
usage: $_selfie COMMAND

Run Syncopy's testing pipeline via SLURM

Arguments:
  COMMAND
    pytest        perform testing using pytest in current user environment
                  (if SLURM is available, tests are executed via `srun`)
            full  (OPTIONAL) if provided, an exhaustive test-run is conducted
                  including, e.g., all selection permutations etc. Default: off
    tox           use tox to set up a new virtual environment (as defined in tox.ini)
                  and run tests within this newly created env
    -h or --help  show this help message and exit
Example:
  $_selfie pytest
  $_selfie pytest full
"
}

# Running this script w/no arguments displays the above help message
if [ "$1" == "" ]; then
    usage
fi

# Set up "global" pytest options for running test-suite (coverage is only done in local pytest runs)
export PYTEST_ADDOPTS="--color=yes --tb=short --verbose"

# The while construction allows parsing of multiple positional/optional args (future-proofing...)
while [ "$1" != "" ]; do
    case "$1" in
        pytest)
            if [ "$2" == "full" ]; then
                fulltests="--full"
            else
                fulltests=""
            fi
            shift
            export PYTHONPATH=$(cd ../../ && pwd)
            if [ $_useSLURM ]; then
                CMD="srun -p DEV --mem=8000m -c 4 pytest $fulltests"
            else
                PYTEST_ADDOPTS="$PYTEST_ADDOPTS --cov=../../syncopy --cov-config=../../.coveragerc"
                export PYTEST_ADDOPTS
                CMD="pytest $fulltests"
            fi
            echo ">>>"
            echo ">>> Running $CMD $PYTEST_ADDOPTS"
            echo ">>>"
            ${CMD}
            ;;
        tox)
            shift
            if [ $_useSLURM ]; then
                CMD="srun -p DEV --mem=8000m -c 4 tox"
            else
                CMD="tox"
            fi
            echo ">>>"
            echo ">>> Running $CMD "
            echo ">>>"
            ${CMD}
            ;;
        -h | --help)
            shift
            usage
            ;;
        *)
            shift
            echo "$_ppname invalid argument '$1'"
            ;;
    esac
done
