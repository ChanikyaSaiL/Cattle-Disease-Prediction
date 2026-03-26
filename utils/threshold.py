import numpy as np
from sklearn.metrics import f1_score

def find_best_thresholds(y_true, y_probs):
    thresholds = []

    for i in range(y_true.shape[1]):
        best_t = 0.5
        best_f1 = 0

        for t in np.linspace(0.1, 0.9, 50):
            preds = (y_probs[:, i] > t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds.append(best_t)

    return np.array(thresholds)