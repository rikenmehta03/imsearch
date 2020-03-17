echo "Installing nmslib"
sudo apt-get install python3-dev
pip install --no-binary :all: nmslib

echo "Installing imsearch"
pip install .