#!/bin/bash
# Some quick shortcuts to ease running Syncopy's testing pipeline on the ESI cluster

# First and foremost, check if `srun` is available
if ! command -v srun &> /dev/null
then
    echo "srun is not available. Are you running me on a SLURM node?"
    exit
fi

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

# The while construction allows parsing of multiple positional/optional args (future-proofing...)
while [ "$1" != "" ]; do
    case "$1" in
        pytest)
            shift
            export PYTHONPATH=$(cd ../../ && pwd)
            export PYTEST_ADDOPTS="--color=yes -q --tb=short -v"
            echo $PYTHONPATH
            echo $PYTEST_ADDOPTS
            srun -p DEV --mem=8000m -c 4 pytest
            ;;
        tox)
            shift
            srun -p DEV --mem=8000m -c 4 tox -r
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
