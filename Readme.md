# Commands used

git init
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
pip install dvc
mkdir data

dvc init
dvc remote add -d myremote gs://mlops-week1-operating-edge-473204-j5/week4/data

gsutil cp gs://mlops-week1-operating-edge-473204-j5/training_data/raw/iris.csv ./data/
gsutil cp gs://mlops-week1-operating-edge-473204-j5/scripts/train.py .
python train.py
dvc add data/ model.joblib train.log
git add data.dvc model.joblib.dvc train.log.dvc train.py .gitignore requirements.txt
git commit -m "trained first batch"
git tag -a "v1.0" -m "trained first batch"
dvc push

# setup Github and push
Go to github and create a new repo
copy name - https://github.com/mlops21f2000513-git/week4-1.git and add origin
git remote add origin https://github.com/mlops21f2000513-git/week4-1.git
git remote -v                           (to check origin)
create token
git push -u origin master

generate secret key to access bucket from IAM 
Add GCP bucket secret key to Actions in github

Add tests
Add yml and dvc config in it
commit and push again to master
create another branch 

# add inference files
gsutil cp gs://mlops-week1-operating-edge-473204-j5/training_data/v2/iris_inference.csv ./data/
gsutil cp gs://mlops-week1-operating-edge-473204-j5/scripts/inference.py .
dvc add data
dvc push
git add data.dvc inference.py

git checkout -b dev
git commit -m "inference data"
git push -u origin dev


# Add yml workflow
mkdir .github
mkdir .github/workflow
add yml config to .github/workflow/runtest.yml


# Files
requirements.txt    -> python packages required to run the scripts
train.py            -> initializes the training. creates a decision tree classifier, trains the model and saves the model.
inference.py        -> load the previously saved model and performs inference on a dataset

iris.csv            -> model trained on this dataset first
iris_inference.csv  -> performed inference on this dataset

model.joblib        -> latest saved model
predictions.csv     -> inference script predictions
train.log           -> log file

.github/workflows/runtest.yml   -> yml file for github actions CI
test.py             -> test file having setup, data validation and model evaluation test cases
