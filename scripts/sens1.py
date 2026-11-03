import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier

d = pd.read_csv("data/model_ready.csv")
y = d["landslide"]
X = d.drop(columns=["landslide", "Latitude", "Longitude"])
groups = (np.floor(d.Latitude / 0.1).astype(int).astype(str) + "_" +
          np.floor(d.Longitude / 0.1).astype(int).astype(str))
cv = GroupKFold(n_splits=5)

best = json.load(open("models/best_params.json"))["Random Forest"]
lc = [c for c in X.columns if c.startswith("lc_")]

sets = {
    "All features": list(X.columns),
    "Without land cover": [c for c in X.columns if c not in lc],
    "Without slope": [c for c in X.columns if c != "slope_deg"],
    "Slope only": ["slope_deg"],
}

rows = []
for name, cols in sets.items():
    m = RandomForestClassifier(n_estimators=300, random_state=42, **best)
    s = cross_validate(m, X[cols], y, groups=groups, cv=cv, scoring=["roc_auc", "recall"])
    rows.append({
        "feature set": name,
        "n_features": len(cols),
        "roc_auc": f"{s['test_roc_auc'].mean():.3f} +/- {s['test_roc_auc'].std():.3f}",
        "recall": f"{s['test_recall'].mean():.3f} +/- {s['test_recall'].std():.3f}",
    })
    print(name, "done")

res = pd.DataFrame(rows)
print("\n", res.to_string(index=False))
res.to_csv("models/sensitivity_results.csv", index=False)