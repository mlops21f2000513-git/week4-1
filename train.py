MODEL_ARTIFACT="model.joblib"
TRAINING_DATA="data/iris.csv"
LOG_FILE="train.log"

import logging

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

logging.info(" *** start training *** ")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = pd.read_csv(TRAINING_DATA)

train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)
prediction = mod_dt.predict(X_test)
accuracy = metrics.accuracy_score(prediction, y_test)
logging.info(f'The accuracy of the Decision Tree is {accuracy:.3f}')

import joblib
joblib.dump(mod_dt, MODEL_ARTIFACT)
logging.info("model saved")

logging.info(" *** training complete *** ")
