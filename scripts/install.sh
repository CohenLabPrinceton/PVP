#!/usr/bin/env bash

sudo apt-get install libatlas-base-dev gfortran

# Install PySide2
# https://bugreports.qt.io/browse/PYSIDE-802
pip install virtualenv

virtualenv venv

source venv/bin/activate

pip install -e .
pip install jupyter ipython tqdm matplotlib external/*.whl
sudo systemctl enable pigpiod
sudo systemctl start pigpiod

pushd lung
pip install -e .
popd
