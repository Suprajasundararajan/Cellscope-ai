import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from .metrics import nll


def equal_vote(P):
    return np.mean(P, 0)


def search_weights(P_val, y_val, step=0.1):
    """Grid-search simplex weights minimising validation NLL (in-domain val only)."""
    n, best = len(P_val), (1e9, None)
    for w in itertools.product(np.arange(0, 1 + 1e-9, step), repeat=n - 1):
        if sum(w) > 1 + 1e-9:
            continue
        w = np.array([*w, 1 - sum(w)])
        score = nll(np.tensordot(w, P_val, 1), y_val)
        if score < best[0]:
            best = (score, w)
    return best[1]


def weighted_vote(P, w):
    return np.tensordot(w, P, 1)


def _feat(P):
    return np.hstack([np.log(p + 1e-8) for p in P])


def fit_stacker(P_val, y_val, k=5, C=1.0):
    """Meta-learner on held-out validation predictions (base models never trained on val -> leakage-safe).
    Also returns 5-fold out-of-fold val probabilities, used for honest temperature fitting."""
    X = _feat(P_val)
    clf = LogisticRegression(C=C, max_iter=3000)
    oof = np.zeros((len(y_val), k))
    raw = cross_val_predict(LogisticRegression(C=C, max_iter=3000), X, y_val, cv=5, method="predict_proba")
    oof[:, np.unique(y_val)] = raw
    clf.fit(X, y_val)
    return clf, oof


def stack_predict(clf, P, k=5):
    out = np.zeros((P[0].shape[0], k)); out[:, clf.classes_] = clf.predict_proba(_feat(P)); return out
