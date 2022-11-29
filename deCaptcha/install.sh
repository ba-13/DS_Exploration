#!/bin/bash

pip install virtualenv
python3 -m venv ./.venv
source ./.venv/bin/activate
pip install numpy pandas matplotlib tensorflow numba opencv-contrib-python tqdm