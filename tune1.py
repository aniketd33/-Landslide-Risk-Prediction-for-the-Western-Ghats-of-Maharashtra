import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

d = pd.read_csv("data/model_ready.csv")
y = d["landslide"]
X = d.drop(columns=["landslide", "Latitude", "Longitude"])
groups = (np.floor(d.Latitude / 0.1).astype(int).astype(str) + "_" +
          np.floor(d.Longitude / 0.1).astype(int).astype(str))
cv = GroupKFold(n_splits=5)

setups = {
    "Logistic Regression": (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        {"logisticregression__C": [0.01, 0.1, 1, 10]}),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [3, 4, 5, 7, 10], "min_samples_leaf": [1, 5, 20]}),
    "Random Forest": (
        RandomForestClassifier(n_estimators=300, random_state=42),
        {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [1, 5, 10]}),
}

# Settings used before tuning (same as Step 10)
before_models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
}

rows, best = [], {}
for name, (model, grid) in setups.items():
    before = cross_val_score(before_models[name], X, y, groups=groups,
                             cv=cv, scoring="roc_auc").mean()
    gs = GridSearchCV(model, grid, cv=cv, scoring="roc_auc")
    gs.fit(X, y, groups=groups)
    best[name] = {k: (v.item() if hasattr(v, "item") else v) for k, v in gs.best_params_.items()}
    rows.append({"model": name,
                 "roc_auc_before": round(before, 3),
                 "roc_auc_after": round(gs.best_score_, 3),
                 "best_settings": str(best[name])})
    print(name, "done")

res = pd.DataFrame(rows)
print("\n", res.to_string(index=False))
res.to_csv("models/tuning_results.csv", index=False)
json.dump(best, open("models/best_params.json", "w"), indent=2)