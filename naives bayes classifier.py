from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# create Gaussian Naive Bayes classifier and train it
nb = GaussianNB()
nb.fit(X_train, y_train)

# make predictions on test set
y_pred = nb.predict(X_test)

# plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([0,1], ["Negative", "Positive"])
plt.yticks([0,1], ["Negative", "Positive"])
plt.title("Confusion Matrix for Naive Bayes Classifier")
plt.show()

# print classification report
cr = classification_report(y_test, y_pred)
print(cr)
