
from sklearn.linear_model import LinearRegression

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

    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=test_ratio)
    
    reg = LinearRegression().fit(x_train, y_train)

    test_score = reg.score(x_test, y_test)

    return test_score
    

