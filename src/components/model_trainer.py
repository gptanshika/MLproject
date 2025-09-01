import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRFRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Split traing and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models ={
                "Linear Regression" : LinearRegression(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "Gradientboost" : GradientBoostingRegressor(),
                "Adaboost" : AdaBoostRegressor(),
                "SVR" : SVR(),
                "Kneighbour" : KNeighborsRegressor(),
                "Xgboost" : XGBRFRegressor()

            }
    


            regressors = {
                "LinearRegression": {
                    "model": LinearRegression(),
                    "params": {
                        "fit_intercept": [True, False],
                        "copy_X": [True, False]
                    }
                },
                "DecisionTree": {
                    "model": DecisionTreeRegressor(),
                    "params": {
                        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                        "splitter": ["best", "random"],
                        "max_depth": [None, 5, 10, 20, 50],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 5, 10]
                    }
                },
                "RandomForest": {
                    "model": RandomForestRegressor(),
                    "params": {
                        "n_estimators": [100, 200, 500],
                        "criterion": ["squared_error", "absolute_error", "friedman_mse"],
                        "max_depth": [None, 5, 10, 20],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 5],
                        "bootstrap": [True, False]
                    }
                },
                "GradientBoosting": {
                    "model": GradientBoostingRegressor(),
                    "params": {
                        "n_estimators": [100, 200, 500],
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "max_depth": [3, 5, 10],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 5]
                    }
                },
                "AdaBoost": {
                    "model": AdaBoostRegressor(),
                    "params": {
                        "n_estimators": [50, 100, 200, 500],
                        "learning_rate": [0.01, 0.05, 0.1, 1.0],
                        "loss": ["linear", "square", "exponential"]
                    }
                },
                "SVR": {
                    "model": SVR(),
                    "params": {
                        "kernel": ["linear", "poly", "rbf", "sigmoid"],
                        "C": [0.1, 1, 10, 100],
                        "gamma": ["scale", "auto"],
                        "degree": [2, 3, 4]
                    }
                },
                "KNN": {
                    "model": KNeighborsRegressor(),
                    "params": {
                        "n_neighbors": [3, 5, 7, 10, 15],
                        "weights": ["uniform", "distance"],
                        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                        "p": [1, 2]  # Manhattan (1), Euclidean (2)
                    }
                },
                "XGBoost": {
                    "model": XGBRFRegressor(eval_metric="rmse", use_label_encoder=False),
                    "params": {
                        "n_estimators": [100, 200, 500],
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "max_depth": [3, 5, 7, 10],
                        "subsample": [0.6, 0.8, 1.0],
                        "colsample_bytree": [0.6, 0.8, 1.0]
                    }
                }
            }

            

            model_report : dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=regressors)

            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<=0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both train and test dataset and the model name is {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)



