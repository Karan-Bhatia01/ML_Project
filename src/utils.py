import os
import sys
import dill
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
import logging

# Save Object
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# Evaluate Models
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            # Check if hyperparameters exist in the params dictionary
            if param.get(model_name):  
                try:
                    # Perform GridSearchCV to tune model hyperparameters
                    gs = GridSearchCV(model, param[model_name], cv=3)
                    gs.fit(X_train, y_train)
                    model.set_params(**gs.best_params_)  # Set best found parameters
                except Exception as e:
                    logging.warning(f"GridSearchCV failed for {model_name}: {str(e)}")
            
            # Fit the model (directly if GridSearchCV fails or no params)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the model's performance score on the test set
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

# Load Object
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
