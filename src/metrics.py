import numpy as np
from scipy.optimize import minimize_scalar


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def nll(p, y):
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def confusion(y, pred, k):
    return np.bincount(y * k + pred, minlength=k * k).reshape(k, k)


def macro_f1(y, pred, k=5):
    cm = confusion(y, pred, k); tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.divide(2 * tp, den, out=np.zeros(k), where=den > 0)))


def per_class_recall(y, pred, k=5):
    cm = confusion(y, pred, k); s = cm.sum(1)
    return np.divide(np.diag(cm), s, out=np.full(k, np.nan), where=s > 0)


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1); acc = pred == y
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def fit_temperature(logp, y):
    r = minimize_scalar(lambda T: nll(softmax(logp / T), y), bounds=(0.05, 10), method="bounded")
    return float(r.x)


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(1)


def mutual_info(member_probs):
    """Ensemble disagreement: H(mean p) - mean H(p_m)."""
    mean = np.mean(member_probs, 0)
    return entropy(mean) - np.mean([entropy(p) for p in member_probs], 0)


def bootstrap_ci(y, pred, fn=macro_f1, n=500, seed=0):
    rng = np.random.default_rng(seed); N = len(y)
    v = []
    for _ in range(n):
        i = rng.integers(0, N, N); v.append(fn(y[i], pred[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
