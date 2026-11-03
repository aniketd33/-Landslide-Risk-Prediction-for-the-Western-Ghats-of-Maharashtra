import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score

d = pd.read_csv("data/model_ready.csv")
y = d["landslide"]
X = d.drop(columns=["landslide", "Latitude", "Longitude"])
groups = (np.floor(d.Latitude / 0.1).astype(int).astype(str) + "_" +
          np.floor(d.Longitude / 0.1).astype(int).astype(str))
cv = GroupKFold(n_splits=5)
best = json.load(open("models/best_params.json"))

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ).set_params(**best["Logistic Regression"]),
    "Decision Tree": DecisionTreeClassifier(random_state=42, **best["Decision Tree"]),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42,
                                            **best["Random Forest"]),
}

# 1. ROC curves from out-of-fold predictions (every point is tested on unseen blocks)
plt.figure(figsize=(6.5, 6))
for name, m in models.items():
    p = cross_val_predict(m, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y, p)
    plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False alarm rate")
plt.ylabel("Landslide places found (recall)")
plt.title("ROC curves (spatial cross-validation)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/roc_curves.png", dpi=200)
plt.close()

# 2. Feature importance (Random Forest fitted on all data)
rf = models["Random Forest"].fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
print("Feature importance:\n", imp.sort_values(ascending=False).round(3))
imp.to_csv("figures/feature_importance.csv", header=["importance"])

plt.figure(figsize=(7, 5))
imp.plot(kind="barh", color="#2E5597")
plt.xlabel("Importance")
plt.title("Random Forest: important factors")
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=200)
print("\nSaved: figures/roc_curves.png and figures/feature_importance.png")