
# Step 1: Set up the environment.

# The basic environment setup can be handled via CVMFS (for now).
# This will give us things like pyLCIO and (Py)ROOT, but not packages
# such as numpy.
source /cvmfs/muoncollider.cern.ch/release/2.9/setup.sh

## To handle numpy, we will use a virtual environment
## in which we can install it via pip.
python -m venv mucol
source mucol/bin/activate
pip install numpy

# Step 2: unpack things!
tar -xzf payload.tar.gz

# Step 3: run things!
outputFile="output.root"
python slcio_analyzer.py \
  -i inputs.txt \
  -n -1 \
  -m "ROOT" \
  -o $outputFile

# Step 4: copy output to output directory

