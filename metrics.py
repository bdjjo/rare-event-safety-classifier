"""Exact capacity ranking is distinct from frozen-threshold alerting."""
import hashlib
import math
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss


def rank_indices(scores, ids):
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(ids) or not np.isfinite(scores).all():
        raise ValueError("Finite scores and aligned IDs required")
    if len(set(ids)) != len(ids):
        raise ValueError("IDs must be unique")
    # Label-blind stable pseudorandom tie breaking, invariant to input order.
    tie = np.array([hashlib.sha256(str(i).encode()).hexdigest() for i in ids])
    return np.lexsort((tie, -scores))


def wilson(successes, total):
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    d = 1 + z*z / total
    c = (p + z*z/(2*total)) / d
    h = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / d
    return [max(0., c-h), min(1., c+h)]


def budget_metrics(y, scores, ids, budget, order=None):
    y = np.asarray(y, dtype=int)
    if not 0 < budget <= 1 or len(y) != len(scores) or not len(y):
        raise ValueError("Invalid budget or data length")
    k = max(1, int(math.floor(len(y) * budget)))
    order = rank_indices(scores, ids) if order is None else order
    tp = int(y[order[:k]].sum())
    positives = int(y.sum())
    recall = tp / positives if positives else None
    return {"budget": budget, "k": k, "tp": tp, "fp": k-tp, "fn": positives-tp,
            "precision": tp/k, "recall": recall,
            "lift": (tp/k)/(positives/len(y)) if positives else None,
            "recall_wilson95": wilson(tp, positives), "precision_wilson95": wilson(tp, k)}


def reliability(y, p, bins=15):
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p*bins).astype(int), bins-1)
    rows = []
    for i in range(bins):
        mask = idx == i
        if mask.any():
            rows.append({"lower": i/bins, "upper": (i+1)/bins, "count": int(mask.sum()),
                         "mean_score": float(p[mask].mean()), "event_rate": float(y[mask].mean())})
    return rows


def evaluate(y, p, ids, budgets, threshold, ranking_scores=None):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite in [0, 1]")
    if set(y) != {0, 1}:
        raise ValueError("Evaluation needs both classes")
    rows = reliability(y, p)
    order = rank_indices(p if ranking_scores is None else ranking_scores, ids)
    chosen = p >= threshold
    tp, fp = int(y[chosen].sum()), int((1-y[chosen]).sum())
    # Scores (not rounded probabilities) preserve ranking under sigmoid saturation.
    ranking = p if ranking_scores is None else ranking_scores
    return {"n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
            "average_precision": float(average_precision_score(y, ranking)),
            "roc_auc": float(roc_auc_score(y, ranking)),
            "brier": float(brier_score_loss(y, p)),
            "positive_brier": float(np.mean((1-p[y == 1])**2)),
            "negative_brier": float(np.mean(p[y == 0]**2)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "ece_15": float(sum(r["count"]*abs(r["mean_score"]-r["event_rate"]) for r in rows)/len(y)),
            "mean_probability": float(p.mean()),
            "accuracy_at_half": float(np.mean((p >= .5) == y)),
            "budgets": [budget_metrics(y, ranking, ids, b, order) for b in budgets],
            "frozen_threshold": {"threshold": float(threshold), "alerts": int(chosen.sum()),
                "alert_rate": float(chosen.mean()), "tp": tp, "fp": fp,
                "precision": tp/int(chosen.sum()) if chosen.any() else None,
                "recall": tp/int(y.sum())}, "reliability": rows}


def choose_threshold(p, budget):
    """Policy-split quantile; ties can exceed capacity at deployment."""
    return float(np.quantile(p, 1-budget, method="higher"))
