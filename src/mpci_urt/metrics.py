from sklearn.metrics import accuracy_score

def classification_accuracy(y_true, y_pred):
    """Accuracy metric used in the supplied experimental script."""
    return accuracy_score(y_true, y_pred)
