"""Metrics in CLOCKWISE degrees, matching API output."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

CLASS_ANGLES_CW = (0, 270, 180, 90)


def orientation_metrics(targets, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=[0, 1, 2, 3], zero_division=0)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1, 2, 3])
    order = [0, 3, 2, 1]
    return {
        "accuracy": float(np.mean(np.asarray(targets) == np.asarray(predictions))),
        "macro_f1": float(np.mean(f1)),
        "per_angle_cw": {str(angle): {"precision": float(precision[i]),
                                      "recall": float(recall[i]), "f1": float(f1[i]),
                                      "support": int(support[i])}
                         for i, angle in enumerate(CLASS_ANGLES_CW)},
        "confusion_angles_cw": [0, 90, 180, 270],
        "confusion_matrix": matrix[np.ix_(order, order)].tolist(),
        "confusions_0_180": int(matrix[0, 2] + matrix[2, 0]),
    }
