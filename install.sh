echo "Installing dependencies"
mkdir .tmp && cd .tmp
git clone https://github.com/fizyr/keras-retinanet.git && cd keras-retinanet
pip install .
cd ../../ && rm -r .tmp

echo "Installing imsearch"
pip install .