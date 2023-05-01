
from sklearn.linear_model import LinearRegression


def linear_regression(df,x_cols,y_col,train_test_split,):
    x = df[x_cols]
    y = df[y_col]

    reg = LinearRegression().fit(x, y)

    score = reg.score(x, y)

    return score