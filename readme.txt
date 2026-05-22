==================================================
==    NEURAL NETWORKS FACE RECOGNITION PROJECT  ==
==================================================

==== INSTALLATION ====
# project was built on Python 3.11.15
# anything newer may cause problems with tensorflow
# you can use pyenv to use multiple python versions

# 1. create venv
python -m venv .venv

# 2. activate venv
source .venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt

# (or optional) for MacOS if you want to use GPU to train models
pip install -r requirements-macos.txt

==== RUNNING PROJECT ====
# training scripts are located in the models folder
# models are pre-trained so you dont have to do this again

# run CUSTOM model on webcam via opencv
python3 run-custom.py

# run MOBILENET model on webcam via opencv
python3 run-mobilenet.py
