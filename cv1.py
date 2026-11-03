import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

d = pd.read_csv("data/model_ready.csv")
y = d["landslide"]
X = d.drop(columns=["landslide", "Latitude", "Longitude"])

# Same 0.1 degree blocks; whole blocks stay together in a fold
groups = (np.floor(d.Latitude / 0.1).astype(int).astype(str) + "_" +
          np.floor(d.Longitude / 0.1).astype(int).astype(str))

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
}

cv = GroupKFold(n_splits=5)
scoring = ["precision", "recall", "f1", "roc_auc"]

rows = []
for name, m in models.items():
    s = cross_validate(m, X, y, groups=groups, cv=cv, scoring=scoring)
    row = {"model": name}
    for k in scoring:
        v = s["test_" + k]
        row[k] = f"{v.mean():.3f} +/- {v.std():.3f}"
    rows.append(row)
    print(name, "done")

res = pd.DataFrame(rows)
print("\n", res.to_string(index=False))
res.to_csv("models/cv_results.csv", index=False)