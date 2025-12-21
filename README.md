<strong> **DO NOT DISTRIBUTE OR PUBLICLY POST SOLUTIONS TO THESE LABS. MAKE ALL FORKS OF THIS REPOSITORY WITH SOLUTION CODE PRIVATE. PLEASE REFER TO THE STUDENT CODE OF CONDUCT AND ETHICAL EXPECTATIONS FOR COLLEGE OF INFORMATION TECHNOLOGY STUDENTS FOR SPECIFICS. ** </strong>

# WESTERN GOVERNORS UNIVERSITY

## D683 – ADVANCED AI AND ML

Welcome to Advanced AI and ML!

For specific task instructions and requirements for this assessment, please refer to the course page.
# D683 DRN1 Task 2 - Churn Prediction

## Requirements
- Windows 10/11 (WGU Assessment Lab)
- Python 3.x
- Packages: see requirements.txt
- Hardware: standard lab PC (CPU, 8–16GB RAM)

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Run
python src/preprocess.py
python src/train_and_evaluate.py
python src/cross_validate.py
python src/tune.py

## Outputs
- data/processed/*.csv
- models/*.joblib
- reports/*.txt


### Python Package Requirements (pinned versions)
Install the exact dependencies using `requirements.txt`. Key packages:

- pandas==2.3.3
- numpy==2.3.5
- scikit-learn==1.8.0
- scipy==1.16.3
- matplotlib==3.10.8
- joblib==1.5.3

Full dependency list (including transitive dependencies):
- See `requirements.txt`

### Hardware
- Standard lab PC (CPU, ~8–16GB RAM; no GPU required)


Install exact dependencies:
pip install -r requirements.txt
