import numpy as np


def f05_entity(pred, truth):
    pred, truth = set(pred), set(truth)
    G, k = len(truth), len(pred)
    if G == 0:
        return 1.0 if k == 0 else 0.0
    if k == 0:
        return 0.0
    tp = len(pred & truth)
    return 5.0 * tp / (G + 4.0 * k)


def macro_f05(pred_map, truth_map, ids):
    s = np.array([f05_entity(pred_map.get(i, ()), truth_map.get(i, ())) for i in ids])
    return s.mean(), s
