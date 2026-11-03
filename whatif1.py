import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

d = pd.read_csv("data/model_ready.csv")
y = d["landslide"]
X = d.drop(columns=["landslide", "Latitude", "Longitude"])
groups = (np.floor(d.Latitude / 0.1).astype(int).astype(str) + "_" +
          np.floor(d.Longitude / 0.1).astype(int).astype(str))
cv = GroupKFold(n_splits=5)
best = json.load(open("models/best_params.json"))["Random Forest"]

factors = [1.0, 1.1, 1.2, 1.3]
probs = {f: np.zeros(len(d)) for f in factors}

for tr, te in cv.split(X, y, groups):
    m = RandomForestClassifier(n_estimators=300, random_state=42, **best)
    m.fit(X.iloc[tr], y.iloc[tr])
    for f in factors:
        Xt = X.iloc[te].copy()
        Xt["monsoon_rain_mm"] = Xt["monsoon_rain_mm"] * f
        probs[f][te] = m.predict_proba(Xt)[:, 1]

def risk_class(p):
    return np.digitize(p, [0.4, 0.7])   # 0 = Low, 1 = Moderate, 2 = High

base = risk_class(probs[1.0])
rows = []
for f in factors:
    p = probs[f]
    c = risk_class(p)
    rows.append({
        "rainfall change": f"+{round((f - 1) * 100)}%",
        "mean risk score": round(p.mean(), 3),
        "% Low": round(100 * (c == 0).mean(), 1),
        "% Moderate": round(100 * (c == 1).mean(), 1),
        "% High": round(100 * (c == 2).mean(), 1),
        "places moved to a higher class": int((c > base).sum()),
    })

res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv("figures/whatif_results.csv", index=False)

res.set_index("rainfall change")[["% Low", "% Moderate", "% High"]].plot(
    kind="bar", stacked=True, figsize=(6.5, 4.5),
    color=["#4CAF50", "#FFB300", "#D32F2F"])
plt.ylabel("Share of places (%)")
plt.title("What-if: monsoon rainfall change")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/whatif.png", dpi=200)
print("\nSaved: figures/whatif_results.csv and figures/whatif.png")