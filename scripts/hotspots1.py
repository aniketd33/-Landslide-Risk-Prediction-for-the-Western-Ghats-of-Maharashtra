import numpy as np
import pandas as pd

g = pd.read_csv("data/susceptibility_grid.csv")
slides = pd.read_csv("data/landslides_clean.csv")

# 1. Keep only High-risk cells and group them into 0.05 degree blocks (about 5.5 km)
high = g[g["risk_class"] == "High"].copy()
high["block"] = (np.floor(high.Latitude / 0.05).astype(int).astype(str) + "_" +
                 np.floor(high.Longitude / 0.05).astype(int).astype(str))

hs = high.groupby("block").agg(
    high_cells=("susceptibility", "size"),
    mean_risk=("susceptibility", "mean"),
    Latitude=("Latitude", "mean"),
    Longitude=("Longitude", "mean"),
).reset_index(drop=True)

# 2. Rank: more high-risk cells first, then higher mean risk. Keep the top 20
hs = hs.sort_values(["high_cells", "mean_risk"], ascending=False).head(20).reset_index(drop=True)
hs.insert(0, "rank", hs.index + 1)

# 3. Nearest recorded landslide (for a place name)
def nearest(lat, lon):
    dy = (slides.Latitude - lat) * 111.0
    dx = (slides.Longitude - lon) * 111.0 * np.cos(np.radians(lat))
    dist = np.sqrt(dx ** 2 + dy ** 2)
    i = dist.idxmin()
    return slides.loc[i, "District"], slides.loc[i, "Location"], dist[i]

near = hs.apply(lambda r: nearest(r.Latitude, r.Longitude), axis=1, result_type="expand")
hs["district"], hs["nearest_recorded_location"], hs["distance_km"] = near[0], near[1], near[2].round(1)

hs["area_km2"] = (hs.high_cells * 0.36).round(1)      # each grid cell is about 600 m x 600 m
hs["mean_risk"] = hs.mean_risk.round(3)
hs["Latitude"] = hs.Latitude.round(3)
hs["Longitude"] = hs.Longitude.round(3)

out = hs[["rank", "district", "nearest_recorded_location", "distance_km",
          "Latitude", "Longitude", "high_cells", "area_km2", "mean_risk"]]
print(out.to_string(index=False))
out.to_csv("figures/hotspots.csv", index=False)
print("\nHigh-risk cells in total:", len(high), "| Saved: figures/hotspots.csv")