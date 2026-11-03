import pandas as pd

d = pd.read_csv("data/final_landslide_dataset.csv")

print("Rows and columns:", d.shape)
print("\nColumn types:\n", d.dtypes)
print("\nMissing values:", d.isna().sum().sum())
print("\nTarget counts:\n", d.landslide.value_counts())
print("\nSummary numbers:\n", d.describe().round(1).T)
print("\nLand cover types:\n", d.landcover_name.value_counts())