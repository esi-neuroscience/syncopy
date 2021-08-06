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
    tox           use tox to set up a new virtual environment (as defined in tox.ini)
                  and run tests within this newly created env
    -h or --help  show this help message and exit
Example:
  $_selfie pytest
"
}

# Running this script w/no arguments displays the above help message
if [ "$1" == "" ]; then
    usage
fi

# Set up "global" pytest options for running test-suite
export PYTEST_ADDOPTS="--color=yes --tb=short --verbose --ignore=syncopy/acme"

# The while construction allows parsing of multiple positional/optional args (future-proofing...)
while [ "$1" != "" ]; do
    case "$1" in
        pytest)
            shift
            export PYTHONPATH=$(cd ../../../ && pwd)
            if [ $_useSLURM ]; then
                srun -p DEV --mem=8000m -c 4 pytest
            else
                pytest
            fi
            ;;
        tox)
            shift
            if [ $_useSLURM ]; then
                srun -p DEV --mem=8000m -c 4 tox -r
            else
                tox -r
            fi
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
