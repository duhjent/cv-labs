from read import read_mnist
from imagedescriptors import extract_hog_features_opencv
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

images_train, y_train = read_mnist('./mnist-csv/mnist_train.csv')
images_test, y_test = read_mnist('./mnist-csv/mnist_test.csv')

X_train = extract_hog_features_opencv(images_train)
X_test = extract_hog_features_opencv(images_test)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
disp.plot()
plt.show()
