import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d = pd.read_csv("data/final_landslide_dataset.csv")
num = ["monsoon_rain_mm", "max_daily_rain_mm", "heavy_rain_days",
       "dist_river_m", "elev_m", "slope_deg", "aspect_deg", "landslide"]

corr = d[num].corr().round(2)
print(corr)

plt.figure(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig("figures/correlation.png", dpi=200)
print("\nSaved: figures/correlation.png")

# Land cover share (percent of each group)
share = pd.crosstab(d.landcover_name, d.landslide, normalize="columns").mul(100).round(1)
print("\nLand cover share (%):\n", share)
share.plot(kind="bar", figsize=(8, 4))
plt.ylabel("Percent of group")
plt.tight_layout()
plt.savefig("figures/landcover_share.png", dpi=200)