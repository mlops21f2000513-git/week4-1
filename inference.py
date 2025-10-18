MODEL_ARTIFACT="model.joblib"
INFERENCE_DATA="data/iris_inference.csv"
LOG_FILE="train.log"
PREDICTIONS="predictions.csv"

import logging

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

logging.info(" *** start inference *** ")


import joblib
loaded_model = joblib.load(MODEL_ARTIFACT)

import pandas as pd
from sklearn import metrics

data = pd.read_csv(INFERENCE_DATA)

X_test = data[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = data.species

prediction = loaded_model.predict(X_test)
prediction_df = pd.DataFrame(prediction, columns=['Prediction'])
prediction_df.to_csv(PREDICTIONS, index=False)

accuracy = metrics.accuracy_score(prediction, y_test)
logging.info(f'The accuracy of the Decision Tree is {accuracy:.3f}')

logging.info(" *** inference complete *** ")
