"""
This module defines what will happen in 'stage-1-train-model':

- download dataset;
- pre-process data into features and labels;
- train machine learning model; and,
- save model to cloud stirage (AWS S3).
"""
from datetime import datetime
from urllib.request import urlopen
from typing import Tuple

import boto3 as aws
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Pre-Processing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, cross_val_predict
from collections import defaultdict
from sklearn.metrics import classification_report

# Models
import lightgbm as lg
import xgboost as xg
from sklearn import metrics
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV

# Miscellineous
import os
import warnings




DATA_URL = ('http://62.171.160.10:9000/datasets/money_mart/final%20data%20sales%20details.xlsx?Content-Disposition=attachment%3B%20filename%3D%22money_mart%2Ffinal%20data%20sales%20details.xlsx%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minio%2F20210419%2F%2Fs3%2Faws4_request&X-Amz-Date=20210419T175633Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=c8962429af5576e638856a934dac7ddede255f09f8b8fa9649db6fdc739492e0')
TRAINED_MODEL_AWS_BUCKET = 'scoringapp-ml-pipeline-project'
TRAINED_MODEL_FILENAME = 'app_scoring.joblib'


def main() -> None:
    """Main script to be executed."""
    data = download_dataset(DATA_URL)
    features, labels = pre_process_data(data)
    trained_model = train_model(features, labels)
    persist_model(trained_model)


def download_dataset(url: str) -> pd.DataFrame:
    """Get data from cloud object storage."""
    print(f'downloading training data from {DATA_URL}')
    data_file = urlopen(url)
    # app_train = pd.read_excel('final data sales details.xlsx', index_col = 'Client Account No.')
    return pd.read_excel(data_file, index_col = 'Client Account No.')


def pre_process_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare raw data for model training."""
    # Variables to be considered by the model
    variables  = ['Final branch', 'Sales Details', 'Gender Revised', 'Marital Status', 'HOUSE', 'Loan Type', 'Fund',
                'Loan Purpose', 'Client Type','Client Classification', 'Currency', 'target', 'Highest Sales','Lowest Sales',
                'Age', 'principal_amount']
    # Subset the data
    app_train = app_train.loc[:, variables]

    # Replace the N/a class with class 'missing'
    app_train['Sales Details'] = np.where(app_train['Sales Details'].isnull(), 'no saledetails', app_train['Sales Details'])
    app_train['HOUSE'] = np.where(app_train['HOUSE'].isnull(), 'not specified', app_train['HOUSE'])
    app_train['Client Type'] = np.where(app_train['Client Type'].isnull(), 'not specified', app_train['Client Type'])
    app_train['Marital Status'] = np.where(app_train['Marital Status'].isnull(), 'not specified', app_train['Marital Status'])
    app_train['Gender Revised'] = np.where(app_train['Gender Revised'].isnull(), 'not specified', app_train['Gender Revised'])
    app_train['Client Classification'] = np.where(app_train['Client Classification'].isnull(),
                                                'not specified', app_train['Client Classification'])


    # Subset numerical data
    numerics = ['int16','int32','int64','float16','float32','float64']
    numerical_vars = list(app_train.select_dtypes(include=numerics).columns)
    numerical_data = app_train[numerical_vars]

    # Fill in missing values
    numerical_data = numerical_data.fillna(numerical_data.mean())

    # Subset categorical data
    cates = ['object']
    cate_vars = list(app_train.select_dtypes(include=cates).columns)
    categorical_data = app_train[cate_vars]
    categorical_data = categorical_data.astype(str)
    categorical_data.shape
    # Instantiate label encoder
    le = preprocessing.LabelEncoder()
    categorical_data = categorical_data.apply(lambda col: le.fit_transform(col).astype(str))
    # categorical_data = le.fit_transform(categorical_data.astype(str))

    # Concat the data
    clean_data = pd.concat([categorical_data, numerical_data], axis = 1)
    clean_data.shape

    # Prepare test data for individual predictions
    test_data = clean_data.drop(['target'], axis = 1)
 
   
    return test_data


def log_model_metrics_to_stdout(
    y_actual: np.ndarray,
    y_predicted: np.ndarray
) -> None:
    """Print model evaluation metrics to stdout."""
    time_now = datetime.now().isoformat(timespec='seconds')
    accuracy = balanced_accuracy_score(
        y_actual,
        y_predicted,
        adjusted=True
    )
    f1 = f1_score(
        y_actual,
        y_predicted,
        average='weighted'
    )
    print(f'iris model metrics @{time_now}')
    print(f' |-- accuracy = {accuracy:.3f}')
    print(f' |-- f1 = {f1:.3f}')


def train_model(features: np.ndarray, labels: np.ndarray) -> BaseEstimator:
    """Train ML model."""
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.1,
        stratify=labels,
        random_state=42
    )
    print('training iris decision tree classifier')
    iris_tree_classifier = DecisionTreeClassifier(
        class_weight='balanced',
        random_state=42
    )
    iris_tree_classifier.fit(X_train, y_train)
    test_data_predictions = iris_tree_classifier.predict(X_test)
    log_model_metrics_to_stdout(y_test, test_data_predictions)
    return iris_tree_classifier


def persist_model(model: BaseEstimator) -> None:
    """Put trained model into cloud object storage."""
    dump(model, TRAINED_MODEL_FILENAME)
    try:
        s3_client = aws.client('s3')
        s3_client.upload_file(
            TRAINED_MODEL_FILENAME,
            TRAINED_MODEL_AWS_BUCKET,
            f'models/{TRAINED_MODEL_FILENAME}'
        )
        print(f'model saved to s3://{TRAINED_MODEL_AWS_BUCKET}'
              f'/{TRAINED_MODEL_FILENAME}')
    except Exception:
        print('could not upload model to S3 - check AWS credentials')


if __name__ == '__main__':
    main()
