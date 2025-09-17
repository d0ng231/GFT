from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

def classify_metrics(y_true, y_pred):
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"balanced_acc": bacc, "precision": prec, "recall": rec, "f1": f1}
