import pickle
from os import listdir, path, abort

import cv2
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

SEGMENTED_DIR = path.join(path.dirname(path.abspath(__file__)), 'segmented')
if not path.exists(SEGMENTED_DIR):
    abort("Missing segmented dir.")

# Load the dataset.
samples = listdir(SEGMENTED_DIR)
n_samples = len(samples)
target = []
dataset = []
for i, sample in zip(range(n_samples), samples):
    img = cv2.imread(path.join(SEGMENTED_DIR, sample), cv2.IMREAD_GRAYSCALE)
    target.append(sample.split("_")[0])
    dataset.append(img)
dataset = np.array(dataset).reshape(n_samples, -1)

# Train the model.
classifier = svm.SVC(gamma="scale")
x_train, x_test, y_train, y_test = train_test_split(dataset, target, test_size=0.2)
classifier.fit(x_train, y_train)

# Test the model.
predicted = classifier.predict(x_test)
print(metrics.classification_report(y_test, predicted))
disp = metrics.plot_confusion_matrix(classifier, x_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

# Model persistence
pickle.dump(classifier, open("model.pickle", mode="wb"))
