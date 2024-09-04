import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

def accuracy(predictions, labels):
    return np.mean(np.array(predictions) == np.array(labels))

def confusion_matrix(predictions, labels):
    return sk_confusion_matrix(labels, predictions)
