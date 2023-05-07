# from sklearn.linear_model import LogisticRegression
# from mlxtend.plotting import plot_confusion_matrix
# import matplotlib.pyplot as plt

# # create a logistic regression classifier and fit the model
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)

# # make predictions on the test set
# y_pred = logreg.predict(X_test)

# # plot the confusion matrix
# plot_confusion_matrix(logreg, X_test, y_test)
# plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# create logistic regression classifier and train it

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# make predictions on test set
y_pred = logreg.predict(X_test)

# plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([0,1], ["Negative", "Positive"])
plt.yticks([0,1], ["Negative", "Positive"])
plt.title("Confusion Matrix for Logistic Regression")
plt.show()

# print classification report
cr = classification_report(y_test, y_pred)
print(cr)
