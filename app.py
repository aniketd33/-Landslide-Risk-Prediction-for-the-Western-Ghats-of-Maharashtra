import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import streamlit as st
from scipy.ndimage import gaussian_filter

st.set_page_config(page_title="Landslide Risk | Western Ghats", layout="wide")

BG = "#0E1117"
COLORS = {"Low": "#2E7D32", "Moderate": "#FFB300", "High": "#FF1744"}


@st.cache_data
def load_data():
    grid = pd.read_csv("data/susceptibility_grid.csv")
    slides = pd.read_csv("data/landslides_clean.csv")
    hot = pd.read_csv("figures/hotspots.csv")
    return grid, slides, hot


def draw_map(grid, show, slides, hot, show_slides, show_hot):
    step = 20 / 3600                      # grid spacing in degrees (about 600 m)
    lon0, lon1 = grid.Longitude.min(), grid.Longitude.max()
    lat0, lat1 = grid.Latitude.min(), grid.Latitude.max()
    ix = np.rint((grid.Longitude - lon0) / step).astype(int)
    iy = np.rint((lat1 - grid.Latitude) / step).astype(int)
    code = grid.risk_class.map({"Low": 0, "Moderate": 1, "High": 2}).to_numpy()
    cls_img = np.full((iy.max() + 1, ix.max() + 1), -1)
    cls_img[iy, ix] = code

    rgba = np.zeros(cls_img.shape + (4,))
    for k, (name, alpha) in enumerate([("Low", 0.45), ("Moderate", 0.85), ("High", 1.0)]):
        if name in show:
            rgb = matplotlib.colors.to_rgb(COLORS[name])
            rgba[cls_img == k] = (*rgb, alpha)

    fig, ax = plt.subplots(figsize=(6.2, 8.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    extent = [lon0 - step / 2, lon1 + step / 2, lat0 - step / 2, lat1 + step / 2]
    ax.imshow(rgba, extent=extent, origin="upper", interpolation="nearest")

    if "High" in show:                    # soft red glow around high-risk zones
        glow = gaussian_filter((cls_img == 2).astype(float), sigma=4)
        glow_rgba = np.zeros(cls_img.shape + (4,))
        glow_rgba[..., 0], glow_rgba[..., 1], glow_rgba[..., 2] = 1.0, 0.09, 0.27
        glow_rgba[..., 3] = np.clip(glow * 1.6, 0, 0.55)
        ax.imshow(glow_rgba, extent=extent, origin="upper", interpolation="bilinear", zorder=1.5)

    if show_slides:
        inside = slides[slides.Longitude.between(lon0, lon1) & slides.Latitude.between(lat0, lat1)]
        ax.scatter(inside.Longitude, inside.Latitude, s=5, c="white", alpha=0.8,
                   linewidths=0, zorder=3)
    if show_hot:
        ax.scatter(hot.Longitude, hot.Latitude, s=90, facecolors="none",
                   edgecolors="#00E5FF", linewidths=1.1, zorder=4)
        for _, r in hot.iterrows():
            ax.annotate(str(int(r["rank"])), (r.Longitude, r.Latitude), color="#00E5FF",
                        fontsize=6, ha="center", va="center", zorder=5)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(np.radians(17.9)))
    ax.tick_params(colors="#8B949E", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#30363D")
    ax.set_xlabel("Longitude", color="#8B949E")
    ax.set_ylabel("Latitude", color="#8B949E")

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[c],
                      markersize=7, label=c) for c in ["Low", "Moderate", "High"]]
    if show_slides:
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                              markersize=4, label="Recorded landslide"))
    if show_hot:
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                              markeredgecolor="#00E5FF", markersize=8, label="Top 20 hotspot"))
    leg = ax.legend(handles=handles, loc="lower left", facecolor="#161B22",
                    edgecolor="#30363D", labelcolor="#E6EDF3", fontsize=8)
    leg.get_frame().set_alpha(0.9)
    fig.tight_layout()
    return fig


def page_map(grid, slides, hot):
    st.title("Landslide Risk Map")
    st.caption("Western Ghats of Maharashtra. Relative landslide susceptibility, not a forecast of when a landslide will happen.")

    n_high = int((grid.risk_class == "High").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("High-risk area", f"{n_high * 0.36:,.0f} km²")
    c2.metric("Top hotspots listed", len(hot))
    c3.metric("Recorded landslides", f"{len(slides):,}")

    left, right = st.columns([3, 2])
    with right:
        st.subheader("Map controls")
        show = st.multiselect("Risk classes to show", ["Low", "Moderate", "High"],
                              default=["Low", "Moderate", "High"])
        show_slides = st.checkbox("Show recorded landslides (white dots)", value=False)
        show_hot = st.checkbox("Show top 20 hotspots (numbered rings)", value=True)

        st.subheader("Check a location")
        lat = st.number_input("Latitude", float(grid.Latitude.min()), float(grid.Latitude.max()),
                              17.93, 0.01, format="%.3f")
        lon = st.number_input("Longitude", float(grid.Longitude.min()), float(grid.Longitude.max()),
                              73.52, 0.01, format="%.3f")
        dx = (grid.Longitude - lon) * np.cos(np.radians(lat))
        dy = grid.Latitude - lat
        i = (dx ** 2 + dy ** 2).idxmin()
        dist_km = float(np.sqrt(dx[i] ** 2 + dy[i] ** 2) * 111.0)
        cls = grid.loc[i, "risk_class"]
        st.markdown(
            f"<div style='font-size:1.6rem;font-weight:700;color:{COLORS[cls]}'>{cls} risk</div>",
            unsafe_allow_html=True)
        st.write(f"Risk score: **{grid.loc[i, 'susceptibility']:.2f}** (0 = lowest, 1 = highest)")
        if dist_km > 2:
            st.warning("This point is far from the mapped area, so the score may not be reliable.")

    with left:
        fig = draw_map(grid, show, slides, hot, show_slides, show_hot)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Top 20 hotspots")
    table = hot.rename(columns={
        "rank": "Rank", "district": "District",
        "nearest_recorded_location": "Nearest recorded landslide",
        "distance_km": "Distance (km)", "high_cells": "High-risk cells",
        "area_km2": "Area (km²)", "mean_risk": "Mean risk"})
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Risk map from the first Random Forest model. District is that of the nearest recorded landslide.")


@st.cache_data
def load_results():
    cv = pd.read_csv("models/final_cv_results.csv")
    roc = pd.read_csv("models/roc_curves.csv")
    cms = json.load(open("models/confusion_matrices.json"))
    wi = pd.read_csv("models/whatif_curve.csv")
    tune = pd.read_csv("models/tuning_results.csv")
    sens = pd.read_csv("models/sensitivity_results.csv")
    imp = pd.read_csv("figures/feature_importance.csv", index_col=0)["importance"]
    return cv, roc, cms, wi, tune, sens, imp


def page_models():
    st.title("Models and What-If")
    st.caption("All scores come from 5-fold spatial cross-validation: every place is tested by a model that never saw its map block.")
    try:
        cv, roc, cms, wi, tune, sens, imp = load_results()
    except FileNotFoundError as e:
        st.error(f"Missing file: {e.filename}. Run export1.py first.")
        return

    tab1, tab2, tab3 = st.tabs(["Model comparison", "Important factors", "Rainfall what-if"])

    with tab1:
        st.subheader("Scores (average over 5 folds)")
        table = pd.DataFrame({"Model": cv.model})
        for key, label in [("accuracy", "Accuracy"), ("precision", "Precision"),
                           ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC")]:
            table[label] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(cv[key + "_mean"], cv[key + "_std"])]
        st.dataframe(table, hide_index=True, width="stretch")
        best_name = cv.loc[cv.roc_auc_mean.idxmax(), "model"]
        st.success(f"Best model by ROC-AUC: {best_name}. Recall matters most here, "
                   "because a missed landslide is worse than a false alarm.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("ROC curves")
            st.line_chart(roc, x="fpr",
                          y=["Logistic Regression", "Decision Tree", "Random Forest", "Random guess"],
                          x_label="False alarm rate", y_label="Landslide places found")
            st.caption("The higher and further left the curve, the better the model.")
        with c2:
            st.subheader("Confusion matrix")
            name = st.radio("Model", list(cms), index=list(cms).index(best_name), horizontal=True)
            (tn, fp), (fn, tp) = cms[name]
            st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]],
                                      index=["Actually safe", "Actually landslide"],
                                      columns=["Predicted safe", "Predicted landslide"]),
                         width="stretch")
            st.write(f"Missed landslides: **{fn}** of {fn + tp} ({100 * fn / (fn + tp):.1f}%). "
                     f"False alarms: **{fp}** of {tn + fp} ({100 * fp / (tn + fp):.1f}%).")

        with st.expander("Tuning: before and after"):
            st.dataframe(tune.rename(columns={"model": "Model", "roc_auc_before": "ROC-AUC before",
                                              "roc_auc_after": "ROC-AUC after",
                                              "best_settings": "Best settings"}),
                         hide_index=True, width="stretch")
            st.caption("Best settings were chosen on the same folds that give the scores, "
                       "so the 'after' numbers may be slightly optimistic.")

    with tab2:
        st.subheader("What the Random Forest relies on")
        df = imp.sort_values(ascending=False).reset_index()
        df.columns = ["Factor", "Importance"]
        st.bar_chart(df, x="Factor", y="Importance", horizontal=True, sort="-Importance")
        st.write(f"Slope is the strongest factor ({imp['slope_deg']:.2f}). Monsoon rainfall "
                 f"({imp['monsoon_rain_mm']:.2f}), elevation ({imp['elev_m']:.2f}) and river "
                 f"distance ({imp['dist_river_log']:.2f}) come far behind. "
                 "Importance shows what the model uses. It does not prove cause and effect.")
        st.subheader("Does the model depend on one column?")
        st.dataframe(sens.rename(columns={"feature set": "Feature set", "n_features": "Features",
                                          "roc_auc": "ROC-AUC", "recall": "Recall"}),
                     hide_index=True, width="stretch")
        st.caption("Removing land cover barely changes the score, so the model does not rely on that "
                   "column. Slope alone does most of the work.")

    with tab3:
        st.subheader("What if monsoon rainfall changes?")
        change = st.select_slider("Change in monsoon rainfall", options=[int(v) for v in wi.rain_change_pct],
                                  value=0, format_func=lambda v: f"{v:+d}%")
        row = wi[wi.rain_change_pct == change].iloc[0]
        base = wi[wi.rain_change_pct == 0].iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("High-risk share", f"{row.pct_high:.1f}%",
                 f"{row.pct_high - base.pct_high:+.1f} points", delta_color="inverse")
        b.metric("Mean risk score", f"{row.mean_risk:.3f}",
                 f"{row.mean_risk - base.mean_risk:+.3f}", delta_color="inverse")
        c.metric("Places moved up a class", f"{int(row.moved_up):,}")
        d.metric("Places moved down a class", f"{int(row.moved_down):,}")
        st.line_chart(wi, x="rain_change_pct", y="pct_high",
                      x_label="Change in monsoon rainfall (%)", y_label="High-risk share (%)")
        st.info("The effect is small and not steady. Rainfall is a weak factor in this data "
                "(one year, coarse 25 km grid, landslide records without dates). "
                "This shows how the model reacts to more rain. It is not a forecast of landslides.")


def page_about():
    st.title("About this project")
    st.caption("Predictive Analytics project: Landslide Risk Prediction for the Western Ghats of Maharashtra")

    st.header("Problem")
    st.write("Landslides happen every monsoon in the Western Ghats and damage roads, villages and property. "
             "Authorities cannot watch every slope, so they need to know which places are more landslide-prone. "
             "This project predicts how landslide-prone a place is (Low, Moderate or High) from terrain, rainfall, "
             "land cover and river data. Intended users: district disaster management teams, road authorities "
             "and local planners.")

    st.header("Objectives")
    st.markdown(
        "1. Find which factors really differ between landslide and non-landslide places, using statistical tests.\n"
        "2. Build and compare at least two predictive models.\n"
        "3. Produce a risk map and a list of high-risk hotspots.\n"
        "4. Test a rainfall what-if scenario and give recommendations.")

    st.header("Data")
    st.dataframe(pd.DataFrame({
        "Data": ["Landslide records", "Rainfall", "Land cover", "Rivers", "Terrain"],
        "What it gives": ["Locations of past landslides", "Monsoon rainfall", "Land type",
                          "Distance to nearest main river", "Elevation, slope, aspect"],
        "Source": ["Maharashtra landslide records (1,490 raw rows)", "IMD gridded data, 2025, 0.25 degree grid",
                   "ESA WorldCover 2021, 10 m", "HydroRIVERS", "Copernicus GLO-30 DEM, 30 m"]}),
        hide_index=True, width="stretch")
    st.write("After cleaning: 2,544 points (1,272 landslide and 1,272 non-landslide), no missing values. "
             "Non-landslide points are random places at least 1 km from any recorded landslide.")

    st.header("Method")
    st.dataframe(pd.DataFrame({
        "Phase": ["1. Problem", "2. Data", "3. EDA and statistics", "4. Preprocessing", "5. Modelling",
                  "6. Evaluation", "7. Interpretation", "8. Conclusion"],
        "What was done": [
            "Objectives, users and scope",
            "Five datasets merged, cleaned and checked",
            "Graphs, Mann-Whitney tests, chi-square test, correlation",
            "Outliers checked, VIF used to keep one rainfall column, dummy variables, log and sine/cosine transforms",
            "Logistic Regression, Decision Tree, Random Forest",
            "5-fold spatial cross-validation, tuning, sensitivity test",
            "Feature importance, rainfall what-if, hotspot list, recommendations",
            "Findings, limitations, future scope"]}),
        hide_index=True, width="stretch")

    st.header("Key findings")
    st.markdown(
        "- **Slope is the strongest factor.** The median slope is about 19 degrees at landslide places and "
        "5.6 degrees elsewhere (test effect 0.81).\n"
        "- **Rainfall, elevation and river distance differ only a little**, partly because the rainfall data is weak.\n"
        "- **Random Forest was compared with Logistic Regression and a Decision Tree** under spatial "
        "cross-validation. See the Models page for the scores.\n"
        "- **Removing land cover barely changed the score**, so the model does not depend on that column.\n"
        "- **Top hotspots cluster along the Sahyadri escarpment** around Poladpur, Mahad and Varandh Ghat, "
        "and most lie within about 3 km of a recorded landslide.")

    st.header("Recommendations")
    st.markdown(
        "1. Inspect and monitor the top hotspots first, before each monsoon.\n"
        "2. Stabilise steep slopes next to roads: drainage, retaining walls, slope protection.\n"
        "3. Avoid new construction and vegetation clearing on steep slopes in High-risk zones.\n"
        "4. Do not use this model as a rainfall alert. Pair it with live IMD rainfall warnings.\n"
        "5. Field-check hotspots before spending money. Scores show relative susceptibility, not certainty.\n"
        "6. Record the date of every landslide and collect daily rainfall.")

    st.header("Limitations")
    st.markdown(
        "- Rainfall is from 2025 only, on a coarse 25 km grid, while the landslide records are older and have no dates. "
        "The straight-edged blocks on the map come from this grid.\n"
        "- Non-landslide points are random, not field-checked as safe, and include more cropland than landslide points.\n"
        "- Scores show relative risk, not exact probability or timing.\n"
        "- Landslide records reflect where surveys were done, not necessarily the true spread of risk.\n"
        "- The risk map comes from the first Random Forest model, built with nearly the same factors.\n"
        "- The study area is limited to the terrain coverage (about 16 to 19.8 N and 73 to 74.15 E).")

    st.header("Future scope")
    st.markdown(
        "- Dated landslide records with daily rainfall, for a real rainfall threshold and forecast.\n"
        "- Finer rainfall data and newer landslide records.\n"
        "- Road network and soil data, and field verification of hotspots.")


try:
    grid, slides, hot = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Check the data and figures folders.")
    st.stop()

st.sidebar.title("Landslide Risk")
st.sidebar.caption("Western Ghats of Maharashtra")
page = st.sidebar.radio("Go to", ["Risk Map", "Models and What-If", "About"])

if page == "Risk Map":
    page_map(grid, slides, hot)
elif page == "Models and What-If":
    page_models()
else:
    page_about()