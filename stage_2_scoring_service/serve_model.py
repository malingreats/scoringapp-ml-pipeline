"""
This module defines what will happen in 'stage-2-deploy-scoring-service':

- download ML model and load into memory;
- define ML scoring REST API endpoints; and,
- start service.

When running the script locally, the scoring service can be tested from
the command line using,

curl http://0.0.0.0:5000/iris/v1/score \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

The expected response should be,

{
    "species_prediction":"setosa"
    "probabilities": "setosa=1.0|versicolor=0.0|virginica=0.0",
    "model_info": "DecisionTreeClassifier(class_weight='balanced', random_state=42)"
}
"""
from urllib.request import urlopen
from typing import Dict

import numpy as np
from flask import Flask, jsonify, make_response, request, Response
from joblib import load
from sklearn.base import BaseEstimator

MODEL_URL = 'http://62.171.160.10:9000/datasets/money_mart/XGBoost.sav?Content-Disposition=attachment%3B%20filename%3D%22money_mart%2FXGBoost.sav%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minio%2F20210419%2F%2Fs3%2Faws4_request&X-Amz-Date=20210419T185255Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=16ce3943699378198f30bf15af84d983af378950ea50497d02e0f0e70e64053c'


app = Flask(__name__)


@app.route('/app_scoring/v1/score', methods=['POST'])
def score() -> Response:
    """API endpoint"""
    try:
        prediction = predict(pre_request)  # only one sample
        # recieve data
        request_data = request.json
        # pre-process
        pre_request = preprocessing(request_data)

        # make prediction
        model_output = model_predictions(pre_request)

        # post response
        response_data = jsonify({**model_output, 'model_info': str(model)})
    except Exception as e:
        return {"status": "Error", "message": str(e)}
        
    return make_response(response_data)


def get_model(url: str) -> BaseEstimator:
    """Get model from cloud object storage."""
    model_file = urlopen(url)
    return load(model_file)



def preprocessing(input_data):
    input_data = pd.DataFrame(input_data, index=[0])
    variables  = ['Final branch', 'Sales Details', 'Gender Revised', 'Marital Status', 'HOUSE', 'Loan Type', 'Fund',
                'Loan Purpose', 'Client Type','Client Classification', 'Currency', 'target', 'Highest Sales','Lowest Sales',
                'Age', 'principal_amount']
    # Subset the data

    app_train = input_data.loc[:, variables]


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
    le = LabelEncoder()
    categorical_data = categorical_data.apply(lambda col: le.fit_transform(col).astype(str))
    # categorical_data = le.fit_transform(categorical_data.astype(str))

    # Concat the data
    clean_data = pd.concat([categorical_data, numerical_data], axis = 1)
    clean_data.shape
    # Prepare test data for individual predictions
    test_data = clean_data.drop(['target'], axis = 1)
    # result = test_data.to_json(orient="columns")
    # parsed = json.loads(result)
    # data = json.dumps(parsed, indent=4)  
    return test_data



def model_predictions(input_data):
    input_data = input_data.values
    y_pred = model.predict_proba(input_data).tolist()[0] 
    return  y_pred[0]


if __name__ == '__main__':
    model = get_model(MODEL_URL)
    print(f'loaded model={model}')
    print(f'starting API server')
    app.run(host='0.0.0.0', port=5000)
