import numpy as np
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from mlxtend.evaluate import bias_variance_decomp
import matplotlib
import pylab as plt
from django.core.files.storage import default_storage
# Create sample data
np.random.seed(0)
X = np.random.rand(100, 5)
y = 1 + 2*X[:,0] + 3*X[:,1] + 4*X[:,2]**2 + 5*X[:,3]**3 + 6*X[:,4]**4 + 0.1*np.random.randn(100)

# Split data into train and test sets
split = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Train polynomial regression models with different degrees
degrees = [1, 2, 3, 4, 5, 6]
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
    this_bias, this_var, _ = bias_variance_decomp(model, X_poly_train, y_train, X_poly_test, y_test, loss='mse', num_rounds=200, random_seed=1)
    bias.append(this_bias)
    variance.append(this_var)

matplotlib.use('Agg')
# Plot different types of errors
# plt.figure(figsize=(10, 5))
plt.plot(degrees, train_errors, label='Train')
plt.plot(degrees, test_errors, label='Test')
plt.xlabel('Degree of polynomial')
plt.ylabel('Mean squared error')
plt.title('Different types of errors')
plt.legend()
plt.savefig('errors.png')
# path =  default_storage.save('file.csv',request.data['file'])
# plt.show()

# Plot bias-variance decomposition
plt.figure(figsize=(10, 5))
plt.plot(degrees, bias, label='Bias')
plt.plot(degrees, variance, label='Variance')
plt.xlabel('Degree of polynomial')
plt.ylabel('Error')
plt.title('Bias-Variance Decomposition')
plt.legend()
plt.savefig('errors2.png')
plt.show()

# Plot R-squared score
plt.figure(figsize=(10, 5))
plt.plot(degrees, r2_scores)
plt.xlabel('Degree of polynomial')
plt.ylabel('R-squared score')
plt.title('Model performance with different degrees in polynomial regression')
plt.savefig('errors3.png')
plt.show()