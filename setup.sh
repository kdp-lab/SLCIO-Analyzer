#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# The basic environment setup can be handled via CVMFS (for now).
# This will give us things like pyLCIO and (Py)ROOT, but not packages
# such as numpy.
source /cvmfs/muoncollider.cern.ch/release/2.9/setup.sh

## To handle numpy, we will use a virtual environment
## in which we can install it via pip.
# TODO: Disabling this, since it might not be a good thing to incl. in repo. -Jan
#source ${SCRIPT_DIR}/env/mucol/bin/activate
