import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

d = pd.read_csv("data/model_ready.csv")
y = d["landslide"]
X = d.drop(columns=["landslide", "Latitude", "Longitude"])
groups = (np.floor(d.Latitude / 0.1).astype(int).astype(str) + "_" +
          np.floor(d.Longitude / 0.1).astype(int).astype(str))
folds = list(GroupKFold(n_splits=5).split(X, y, groups))
best = json.load(open("models/best_params.json"))


def make_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ).set_params(**best["Logistic Regression"]),
        "Decision Tree": DecisionTreeClassifier(random_state=42, **best["Decision Tree"]),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42,
                                                **best["Random Forest"]),
    }


# 1. Scores, ROC curves and confusion matrices (tuned models, 5-fold spatial CV)
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
grid = np.linspace(0, 1, 101)
roc_df = pd.DataFrame({"fpr": grid})
rows, cms = [], {}
for name in make_models():
    per = {k: [] for k in metrics}
    oof_prob = np.zeros(len(d))
    oof_pred = np.zeros(len(d), dtype=int)
    for tr, te in folds:
        m = make_models()[name]
        m.fit(X.iloc[tr], y.iloc[tr])
        prob = m.predict_proba(X.iloc[te])[:, 1]
        pred = m.predict(X.iloc[te])
        yt = y.iloc[te]
        per["accuracy"].append(accuracy_score(yt, pred))
        per["precision"].append(precision_score(yt, pred))
        per["recall"].append(recall_score(yt, pred))
        per["f1"].append(f1_score(yt, pred))
        per["roc_auc"].append(roc_auc_score(yt, prob))
        oof_prob[te] = prob
        oof_pred[te] = pred
    row = {"model": name}
    for k in metrics:
        row[k + "_mean"] = float(np.mean(per[k]))
        row[k + "_std"] = float(np.std(per[k]))
    rows.append(row)
    fpr, tpr, _ = roc_curve(y, oof_prob)
    roc_df[name] = np.interp(grid, fpr, tpr)
    cms[name] = confusion_matrix(y, oof_pred).tolist()
    print(name, "done")
roc_df["Random guess"] = grid

res = pd.DataFrame(rows)
res.to_csv("models/final_cv_results.csv", index=False)
roc_df.to_csv("models/roc_curves.csv", index=False)
json.dump(cms, open("models/confusion_matrices.json", "w"))
show = res[["model"]].copy()
for k in metrics:
    show[k] = [f"{m:.3f} +/- {s:.3f}" for m, s in zip(res[k + "_mean"], res[k + "_std"])]
print("\n", show.to_string(index=False))

# 2. Rainfall what-if (tuned Random Forest, each place scored by a model that never saw its block)
changes = [-20, -10, 0, 10, 20, 30, 40, 50]
probs = {c: np.zeros(len(d)) for c in changes}
for tr, te in folds:
    m = RandomForestClassifier(n_estimators=300, random_state=42, **best["Random Forest"])
    m.fit(X.iloc[tr], y.iloc[tr])
    for c in changes:
        Xt = X.iloc[te].copy()
        Xt["monsoon_rain_mm"] = Xt["monsoon_rain_mm"] * (1 + c / 100)
        probs[c][te] = m.predict_proba(Xt)[:, 1]

base = np.digitize(probs[0], [0.4, 0.7])
wi = []
for c in changes:
    cls = np.digitize(probs[c], [0.4, 0.7])
    wi.append({"rain_change_pct": c,
               "mean_risk": float(probs[c].mean()),
               "pct_low": 100 * float((cls == 0).mean()),
               "pct_moderate": 100 * float((cls == 1).mean()),
               "pct_high": 100 * float((cls == 2).mean()),
               "moved_up": int((cls > base).sum()),
               "moved_down": int((cls < base).sum())})
wi = pd.DataFrame(wi)
wi.to_csv("models/whatif_curve.csv", index=False)
print("\n", wi.round(3).to_string(index=False))
print("\nSaved 4 files in models/")