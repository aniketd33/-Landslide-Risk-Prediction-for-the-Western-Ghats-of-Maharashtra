# Landslide Risk Prediction for the Western Ghats of Maharashtra

Predictive Analytics capstone project (6 credits), B.Sc. Data Science, Savitribai Phule Pune University (SPPU), Pune.

A machine learning project that predicts **which locations in the Western Ghats of Maharashtra are most prone to landslides**, shows them on an interactive dashboard, and lets users test a "what if rainfall increases?" scenario.

---

## 1. Problem

Landslides occur every monsoon in the Western Ghats and cause loss of life, damage to roads and homes, and disruption. Authorities have limited resources, so they need to know **where to focus first**.

## 2. What we predict

For any location, the model predicts its **landslide susceptibility**: a risk score from 0 to 1, grouped as **Low (< 0.4), Moderate (0.4 to 0.7) and High (>= 0.7)**.

- This is **binary classification** (landslide = 1, no landslide = 0), and the output is a probability.
- It tells us **where** a landslide is likely, **not when**. The landslide records have no dates, so date forecasting is not possible.

## 3. Objectives

1. Identify the factors that influence landslides.
2. Build and compare at least two machine learning models.
3. Evaluate them fairly using spatial cross-validation.
4. Produce a risk map and a list of high-risk hotspots.
5. Test how risk changes if rainfall increases (what-if analysis).
6. Present everything in an easy-to-use dashboard with recommendations.

## 4. Data

One dataset: `final_landslide_dataset.csv`, with landslide and non-landslide locations.

| Type | Features |
|---|---|
| Rainfall | Monsoon rainfall, maximum daily rainfall, heavy-rain days |
| Terrain | Elevation, slope, aspect (direction the slope faces) |
| Water | Distance to nearest river |
| Land | Land cover class |

Other files used by the dashboard:
- `landslides_clean.csv`: 1,374 recorded landslide locations
- `susceptibility_grid.csv`: 123,519 map cells (about 600 m each) with predicted risk, from a Random Forest model

## 5. Method (follows the 8 faculty phases)

| Phase | What we did |
|---|---|
| 1. Problem definition | Defined the problem, target and objectives (above) |
| 2. Data collection and quality check | Checked shape, data types, missing values and duplicates |
| 3. EDA and statistics | Histograms, boxplots, landslide vs non-landslide comparison, Mann-Whitney U and chi-square tests, correlation, outliers (IQR) |
| 4. Preprocessing | VIF (multicollinearity) check, rare land covers grouped, log of river distance, sine/cosine of aspect, dummy variables for land cover, scaling inside the pipeline, spatial blocks of 0.1 degree |
| 5. Model building | Logistic Regression, Decision Tree, Random Forest |
| 6. Evaluation | 5-fold **spatial** GroupKFold cross-validation, GridSearchCV tuning, sensitivity test |
| 7. Interpretation | Feature importance, rainfall what-if, hotspots, recommendations |
| 8. Conclusion | Findings, limitations, future scope |

**Why spatial cross-validation?** Nearby locations look alike. If neighbours land in both training and test sets, the score looks better than it really is. Spatial folds keep neighbours together, so the model is tested on genuinely new areas.

## 6. Results

Cross-validated scores (mean of 5 spatial folds, after tuning):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.740 | 0.745 | 0.731 | 0.737 | 0.806 |
| Decision Tree | 0.744 | 0.711 | 0.819 | 0.759 | 0.807 |
| **Random Forest (best)** | **0.757** | 0.735 | 0.803 | **0.767** | **0.832** |

Best Random Forest settings: `max_depth = 12`, `min_samples_leaf = 5`.

**Most important factors (Random Forest):**

| Factor | Importance |
|---|---|
| Slope | 0.48 |
| Monsoon rainfall | 0.11 |
| Elevation | 0.11 |
| Distance to river (log) | 0.09 |
| Land cover and aspect | smaller |

**Sensitivity test (ROC-AUC):**

| Feature set | ROC-AUC |
|---|---|
| All features | 0.832 |
| Without land cover | 0.831 |
| Without slope | 0.743 |
| Slope only | 0.772 |

Slope drives the model. Removing land cover changes almost nothing, so the model is not relying on land cover.

**Rainfall what-if (mean risk, share of High-risk area):**

| Rainfall change | Mean risk | % High |
|---|---|---|
| 0% | 0.486 | 29.0 |
| +10% | 0.483 | 27.9 |
| +20% | 0.488 | 28.8 |
| +30% | 0.500 | 30.9 |

The effect of rainfall is small and not steady, because the data has one rainfall value per area (no date-wise rainfall).

**Hotspots:** the 20 largest clusters of High-risk cells are listed in `figures/hotspots.csv` and shown on the dashboard map.

## 7. Dashboard (Streamlit)

Three pages:

1. **Risk Map**: predicted risk map, optional recorded landslides and numbered hotspots, "Check a location" by latitude and longitude, top-20 hotspot table.
2. **Models and What-If**: model comparison, ROC curves, confusion matrices, tuning results, feature importance, sensitivity test, rainfall what-if slider.
3. **About**: problem, objectives, data, method, key findings, recommendations, limitations, future scope.

## 8. Recommendations

- Prioritise inspection and early-warning systems in the listed hotspots.
- Avoid new construction and road cutting on steep, high-risk slopes.
- Use slope as the first screening factor when planning in the Western Ghats.
- Combine this map with real-time rainfall alerts during monsoon.

## 9. Limitations

- It is a susceptibility map, not a forecast of when a landslide will happen.
- Rainfall is one value per area, so its true effect is probably under-estimated.
- Accuracy is moderate (ROC-AUC about 0.83). Use it as decision support, not as a final decision.
- The hotspot district is that of the nearest recorded landslide, not an official boundary.
- Scores can differ by about +/- 0.01 between library versions and folds.

## 10. Future scope

- Add date-wise rainfall and dated landslide records to enable true forecasting.
- Add soil type, geology, and distance to roads.
- Try gradient boosting models and compare.
- Add live rainfall feeds to the dashboard.

## 11. Project structure

```
landslide_app/
├── app.py                      # Streamlit dashboard
├── .streamlit/config.toml      # Theme settings
├── data/
│   ├── final_landslide_dataset.csv
│   ├── model_ready.csv
│   ├── susceptibility_grid.csv
│   └── landslides_clean.csv
├── figures/                    # EDA, ROC, importance, what-if charts, hotspots.csv
├── models/                     # Results tables, best_params.json
└── scripts: check_data.py, eda1-3.py, prep1-3.py, train1.py, cv1.py,
             tune1.py, sens1.py, interp1.py, whatif1.py, hotspots1.py, export1.py
```

## 12. How to run

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows

# 2. Install packages
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels streamlit

# 3. Run analysis scripts in this order
python check_data.py
python eda1.py && python eda2.py && python eda3.py
python prep1.py && python prep2.py && python prep3.py
python train1.py
python cv1.py
python tune1.py
python sens1.py
python interp1.py
python whatif1.py
python hotspots1.py
python export1.py

# 4. Start the dashboard
streamlit run app.py
```

## 13. Tools

Python, pandas, NumPy, scikit-learn, statsmodels, SciPy, matplotlib, seaborn, Streamlit, VS Code.

## 14. Author

Aniket Dombale, Final-year B.Sc. Data Science, Department of Technology, SPPU, Pune.
