========================================
Neural Networks Face Recognition Project
========================================

==== Installation (Python 3.11) ====

# create venv
python -m venv .venv

# activate it
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt


==== Run Project ====
# training scripts are located in the models folder
# models are pre-trained so you dont have to do this again

# run custom model on webcam via opencv
python3 run-custom.py

# run mobilenet model on webcam via opencv
python3 run-mobilenet.py
