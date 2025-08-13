from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def build(C=1.0, penalty="l2", solver="lbfgs", max_iter=500, class_weight="balanced"):
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, penalty=penalty, solver=solver, max_iter=max_iter, class_weight=class_weight
        )),
    ])