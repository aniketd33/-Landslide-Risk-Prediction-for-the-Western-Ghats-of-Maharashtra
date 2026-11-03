import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

d = pd.read_csv("data/final_landslide_dataset.csv")
num = ["monsoon_rain_mm", "max_daily_rain_mm", "heavy_rain_days",
       "dist_river_m", "elev_m", "slope_deg", "aspect_deg"]

# 1. Outliers (IQR rule)
rows = []
for c in num:
    q1, q3 = d[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = int(((d[c] < q1 - 1.5 * iqr) | (d[c] > q3 + 1.5 * iqr)).sum())
    rows.append({"factor": c, "outliers": n, "percent": round(100 * n / len(d), 1)})
out = pd.DataFrame(rows)
print("OUTLIERS\n", out.to_string(index=False))
out.to_csv("figures/outliers.csv", index=False)

# 2. VIF
X = add_constant(d[num])
vif = pd.DataFrame({
    "factor": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
})
vif = vif[vif.factor != "const"].round(2)
print("\nVIF\n", vif.to_string(index=False))
vif.to_csv("figures/vif.csv", index=False)