import pandas as pd
from scipy import stats

d = pd.read_csv("data/final_landslide_dataset.csv")
num = ["monsoon_rain_mm", "max_daily_rain_mm", "heavy_rain_days",
       "dist_river_m", "elev_m", "slope_deg"]

rows = []
for c in num:
    a = d.loc[d.landslide == 1, c]   # landslide places
    b = d.loc[d.landslide == 0, c]   # non-landslide places
    u, p = stats.mannwhitneyu(a, b)
    rows.append({
        "factor": c,
        "median_landslide": round(a.median(), 1),
        "median_no_landslide": round(b.median(), 1),
        "p_value": f"{p:.2e}",
        "effect (0.5 = none)": round(u / (len(a) * len(b)), 2),
        "real_difference": "YES" if p < 0.05 else "NO",
    })

res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv("figures/hypothesis_tests.csv", index=False)

# Land cover: chi-square test
table = pd.crosstab(d.landcover_name, d.landslide)
chi2, p, dof, _ = stats.chi2_contingency(table)
print(f"\nLand cover vs landslide: chi-square = {chi2:.1f}, p = {p:.2e}")