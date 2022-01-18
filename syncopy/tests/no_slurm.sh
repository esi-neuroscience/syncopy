#!/bin/bash
# Sabotage local tox environment so that `sinfo` is not working any more
if [ -n "$NO_SLURM" ]; then
    echo "ofnis" >| $VIRTUAL_ENV/bin/sinfo && chmod a+x $VIRTUAL_ENV/bin/sinfo
fi
