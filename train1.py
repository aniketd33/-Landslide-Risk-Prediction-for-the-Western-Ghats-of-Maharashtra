import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

d = pd.read_csv("data/model_ready.csv")
y = d["landslide"]
X = d.drop(columns=["landslide", "Latitude", "Longitude"])

# Spatial split: 0.1 degree blocks, 25% of blocks kept for testing
block = (np.floor(d.Latitude / 0.1).astype(int).astype(str) + "_" +
         np.floor(d.Longitude / 0.1).astype(int).astype(str))
rng = np.random.RandomState(42)
blocks = block.unique()
test_blocks = rng.choice(blocks, size=int(0.25 * len(blocks)), replace=False)
test = block.isin(test_blocks)

X_tr, X_te, y_tr, y_te = X[~test], X[test], y[~test], y[test]
print(f"Train: {len(X_tr)} | Test: {len(X_te)} | Test landslide share: {y_te.mean():.2f}")

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
}

rows = []
for name, m in models.items():
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    prob = m.predict_proba(X_te)[:, 1]
    rows.append({
        "model": name,
        "precision": round(precision_score(y_te, pred), 3),
        "recall": round(recall_score(y_te, pred), 3),
        "f1": round(f1_score(y_te, pred), 3),
        "roc_auc": round(roc_auc_score(y_te, prob), 3),
    })
    print(f"\n{name} confusion matrix:\n", confusion_matrix(y_te, pred))

res = pd.DataFrame(rows)
print("\n", res.to_string(index=False))
res.to_csv("models/baseline_results.csv", index=False)