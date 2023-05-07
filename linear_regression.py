# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

# # Generate random data
# x = np.random.rand(100, 1) * 10
# y = 2 * x + 1 + np.random.randn(100, 1)

# # Fit linear regression model
# model = LinearRegression()
# model.fit(x, y)

# # Predict values using the model
# y_pred = model.predict(x)

# # Plot the data points and regression line
# plt.scatter(x, y)
# plt.plot(x, y_pred, color='red')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Linear Regression Analysis')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate random data
x = np.random.rand(100, 1) * 10
y = x ** 2 - 5 * x + 1 + np.random.randn(100, 1)

# Fit polynomial regression model
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
model = LinearRegression()
model.fit(x_poly, y)

# Predict values using the model
y_pred = model.predict(x_poly)

# Plot the data points and polynomial regression curve
plt.scatter(x, y)
plt.plot(x,y_pred ,color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression Analysis')
plt.show()
