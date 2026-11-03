import numpy as np
import pandas as pd

d = pd.read_csv("data/final_landslide_dataset.csv")

# 1. Group rare land covers
d["landcover_name"] = d["landcover_name"].replace(
    {"Water": "Other", "Bare/sparse": "Other", "Mangroves": "Other"})

# 2. Transformations
d["dist_river_log"] = np.log1p(d["dist_river_m"])
d["aspect_sin"] = np.sin(np.radians(d["aspect_deg"]))
d["aspect_cos"] = np.cos(np.radians(d["aspect_deg"]))

# 3. Dummy variables (Tree cover is the base, so it gets no column)
dummies = pd.get_dummies(d["landcover_name"], prefix="lc").astype(int)
dummies = dummies.drop(columns="lc_Tree cover")

# 4. Final table (Latitude/Longitude kept only for the spatial split later)
keep = ["Latitude", "Longitude", "monsoon_rain_mm", "dist_river_log",
        "elev_m", "slope_deg", "aspect_sin", "aspect_cos"]
out = pd.concat([d[keep], dummies, d["landslide"]], axis=1)

print("Shape:", out.shape)
print("Missing:", out.isna().sum().sum())
print("\nColumns:\n", list(out.columns))
print("\nLand cover groups:\n", d["landcover_name"].value_counts())

out.to_csv("data/model_ready.csv", index=False)
print("\nSaved: data/model_ready.csv")