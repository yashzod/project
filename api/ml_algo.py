
from sklearn.linear_model import LinearRegression
import sklearn.model_selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from mlxtend.evaluate import bias_variance_decomp

{
    "model":"linearregression",
    "file_name":"file_name",
    "train_test_split":{
        "train":70,
        "test":20,
    },
    "column_processing":[
        {
            "column_name1":{
                "missing_value":"average",
                "encoding":"one_hot",
                "feature_scaling":"0-1"
            },
            "column_name2":{
                "missing_value":"average",
                "encoding":"one_hot",
                "feature_scaling":"-1-1"
            }
        }
    ],
}

def linear_regression(df,x_cols,y_col,train_test_split):

    dataX = df[x_cols]
    dataY = df[y_col]

    train_ratio = train_test_split['train']/100
    test_ratio = train_test_split['test']/100

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataX, dataY, test_size=test_ratio)
    
    # reg = LinearRegression().fit(x_train, y_train)

    # test_score = reg.score(x_test, y_test)

    degrees = list(range(10))

    train_errors, test_errors, r2_scores, bias, variance = [], [], [], [], []

    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)
        X_poly_train = poly_features.fit_transform(X_train)
        X_poly_test = poly_features.transform(X_test)
        model = LinearRegression()
        model.fit(X_poly_train, y_train)
        train_pred = model.predict(X_poly_train)
        test_pred = model.predict(X_poly_test)
        train_error = mean_squared_error(y_train, train_pred)
        test_error = mean_squared_error(y_test, test_pred)
        train_errors.append(train_error)
        test_errors.append(test_error)
        r2_scores.append(r2_score(y_test, test_pred))
        # this_bias, this_var, _ = bias_variance_decomp(model, X_poly_train, y_train, X_poly_test, y_test, loss='mse', num_rounds=200, random_seed=1)
        # bias.append(this_bias)
        # variance.append(this_var)

    data = []

    data.append([
        {
            "x":degrees,
            "train errors":train_errors,
            "test errors":test_errors
        }
    ])

    return data
    

