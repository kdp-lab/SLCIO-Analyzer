
# Arguments:
inputFile=$1
version=$2 # what version ntuples we are running on

# Step 1: Set up the environment.
# # Non-containerized approach -- will not work across all OS options!
# # The basic environment setup can be handled via CVMFS (for now).
# # This will give us things like pyLCIO and (Py)ROOT, but not packages
# # such as numpy.
# source /cvmfs/muoncollider.cern.ch/release/2.9/setup.sh

# ## To handle numpy, we will use a virtual environment
# ## in which we can install it via pip.
# python -m venv mucol
# source mucol/bin/activate
# pip install numpy

echo ">>> HOSTNAME = ${HOSTNAME}"
echo ">>> SINGULARITY_NAME = ${SINGULARITY_NAME}"
echo ">>> Setting up environment:"
echo "source /opt/setup_mucoll.sh --> "
source /opt/setup_mucoll.sh
echo ">>> Setup completed."

# Step 2: unpack things!
tar -xzf payload.tar.gz

# Step 3: run things!
outputFile="output.root"
python slcio_analyzer.py \
  -i $inputFile \
  -n -1 \
  -m "ROOT" \
  -o $outputFile \
  -version $version

