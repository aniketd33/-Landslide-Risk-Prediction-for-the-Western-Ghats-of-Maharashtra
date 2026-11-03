import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d = pd.read_csv("data/final_landslide_dataset.csv")
num = ["monsoon_rain_mm", "max_daily_rain_mm", "heavy_rain_days",
       "dist_river_m", "elev_m", "slope_deg"]

# 1. Table: average and middle value for each group
table = d.groupby("landslide")[num].agg(["mean", "median"]).round(1).T
print(table)
table.to_csv("figures/group_comparison.csv")

# 2. Picture 1: histograms (shape of the data)
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, c in zip(axes.ravel(), num):
    sns.histplot(data=d, x=c, hue="landslide", bins=30, stat="density",
                 common_norm=False, element="step", ax=ax)
    ax.set_title(c)
plt.tight_layout()
plt.savefig("figures/distributions.png", dpi=200)
plt.close()

# 3. Picture 2: boxplots (spread and outliers)
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, c in zip(axes.ravel(), num):
    sns.boxplot(data=d, x="landslide", y=c, ax=ax)
    ax.set_title(c)
plt.tight_layout()
plt.savefig("figures/boxplots.png", dpi=200)
plt.close()

print("\nSaved: figures/distributions.png and figures/boxplots.png")