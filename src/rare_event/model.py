"""Sparse TF-IDF logistic baselines and disjoint sigmoid calibration."""
import re
import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def clean_text(text):
    return re.sub(r"\s*\[ticket=[^\]]*\]", "", text).lower()


class SafetyClassifier:
    def __init__(self, kind="word_balanced", seed=17):
        if kind not in {"word_balanced", "word_unweighted", "char_balanced"}:
            raise ValueError(f"Unknown model {kind}")
        self.kind = kind
        self.seed = seed
        char = kind.startswith("char")
        self.vectorizer = TfidfVectorizer(
            preprocessor=clean_text, analyzer="char_wb" if char else "word",
            ngram_range=(3, 5) if char else (1, 2), min_df=2,
            max_features=30000, sublinear_tf=True)
        self.classifier = LogisticRegression(
            C=1.0, class_weight=None if kind.endswith("unweighted") else "balanced",
            solver="liblinear", max_iter=1000, random_state=seed)
        self.calibrator = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000)

    def fit(self, train_text, train_y, calibration_text, calibration_y):
        x = self.vectorizer.fit_transform(train_text)
        self.classifier.fit(x, train_y)
        scores = self.decision_function(calibration_text).reshape(-1, 1)
        # Preserve deployment-like prevalence: no weighting/oversampling here.
        self.calibrator.fit(scores, calibration_y)
        if self.calibrator.coef_[0, 0] <= 0:
            raise ValueError("Calibration slope must be positive for rank preservation")
        return self

    def decision_function(self, texts):
        return self.classifier.decision_function(self.vectorizer.transform(texts))

    def predict_from_scores(self, scores, calibrated=True):
        scores = np.asarray(scores)
        if calibrated:
            return self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        return expit(scores)

    def predict_proba(self, texts, calibrated=True):
        return self.predict_from_scores(self.decision_function(texts), calibrated)

    def top_features(self, n=20):
        terms = self.vectorizer.get_feature_names_out()
        weights = self.classifier.coef_[0]
        order = np.argsort(weights)
        return {"risk": [{"term": terms[i], "weight": float(weights[i])} for i in order[-n:][::-1]],
                "benign": [{"term": terms[i], "weight": float(weights[i])} for i in order[:n]]}
