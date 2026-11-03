import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

d = pd.read_csv("data/final_landslide_dataset.csv")

def show_vif(cols, title):
    X = add_constant(d[cols])
    v = pd.DataFrame({"factor": X.columns,
                      "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
    print(title, "\n", v[v.factor != "const"].round(2).to_string(index=False), "\n")

# Try 1: drop heavy_rain_days
show_vif(["monsoon_rain_mm", "max_daily_rain_mm", "dist_river_m",
          "elev_m", "slope_deg", "aspect_deg"], "After dropping heavy_rain_days")

# Try 2: keep only monsoon_rain_mm as the rain column
show_vif(["monsoon_rain_mm", "dist_river_m",
          "elev_m", "slope_deg", "aspect_deg"], "Only monsoon_rain_mm kept")