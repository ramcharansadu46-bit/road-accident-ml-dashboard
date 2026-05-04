import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline

from scipy.stats import shapiro, ttest_1samp, norm, chisquare


# chart colors for dark background
BG     = "#0b0f1a"
CARD   = "#1a2236"
RED    = "#e84545"
ORANGE = "#f5a623"
GREEN  = "#3fcf8e"
BLUE   = "#4a9eff"
PURPLE = "#b06dff"
GRAY   = "#7a8aaa"
WHITE  = "#e8edf5"

CHART_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def set_dark_theme():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    CARD,
        "axes.edgecolor":    "#242f45",
        "axes.labelcolor":   GRAY,
        "axes.titlecolor":   WHITE,
        "xtick.color":       GRAY,
        "ytick.color":       GRAY,
        "text.color":        WHITE,
        "grid.color":        "#1f2d45",
        "grid.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.titlesize":    13,
        "axes.labelsize":    10,
    })


def save_chart(fig, filename):
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"/static/charts/{filename}"


# ----- DATA LOADING -----

def load_data(path):
    df = pd.read_csv(path)

    year_cols = [c for c in df.columns if c.isdigit()]

    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # fill NaN with 0
    # Telangana districts have NaN for 2003-2013 because the state was
    # created in 2014 (carved out of Andhra Pradesh). Row mean would wrongly
    # inflate pre-2014 values, so we fill with 0 instead.
    df[year_cols] = df[year_cols].fillna(0)

    df[year_cols] = df[year_cols].round(0).astype(int)
    return df


def get_national_totals(df):
    year_cols = [c for c in df.columns if c.isdigit()]
    totals = df[year_cols].sum()
    years = np.array([int(y) for y in year_cols])
    accidents = totals.values.astype(float)
    return years, accidents


# ----- MODEL TRAINING -----

def train_all_models(years, accidents):
    X = years.reshape(-1, 1)
    y = accidents

    # last 3 years go to test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3, shuffle=False)

    results = {}

    # linear regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    lr_train_pred = lr_model.predict(X_train)
    lr_test_pred  = lr_model.predict(X_test)
    lr_all_pred   = lr_model.predict(X)

    results["linear"] = {
        "model":    lr_model,
        "pred_all": lr_all_pred,
        "r2_train": r2_score(y_train, lr_train_pred),
        "r2_test":  r2_score(y_test, lr_test_pred),
        "r2":       r2_score(y, lr_all_pred),
        "rmse":     np.sqrt(mean_squared_error(y_test, lr_test_pred)),
        "mae":      mean_absolute_error(y_test, lr_test_pred),
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
    }

    # polynomial regression degree 3
    poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    poly_model.fit(X_train, y_train)

    poly_train_pred = poly_model.predict(X_train)
    poly_test_pred  = poly_model.predict(X_test)
    poly_all_pred   = poly_model.predict(X)

    results["polynomial"] = {
        "model":    poly_model,
        "pred_all": poly_all_pred,
        "r2_train": r2_score(y_train, poly_train_pred),
        "r2_test":  r2_score(y_test, poly_test_pred),
        "r2":       r2_score(y, poly_all_pred),
        "rmse":     np.sqrt(mean_squared_error(y_test, poly_test_pred)),
        "mae":      mean_absolute_error(y_test, poly_test_pred),
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
    }

    # random forest
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred  = rf_model.predict(X_test)
    rf_all_pred   = rf_model.predict(X)

    results["random_forest"] = {
        "model":    rf_model,
        "pred_all": rf_all_pred,
        "r2_train": r2_score(y_train, rf_train_pred),
        "r2_test":  r2_score(y_test, rf_test_pred),
        "r2":       r2_score(y, rf_all_pred),
        "rmse":     np.sqrt(mean_squared_error(y_test, rf_test_pred)),
        "mae":      mean_absolute_error(y_test, rf_test_pred),
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
    }

    return results


def predict_future_years(models, future_years):
    X_future = future_years.reshape(-1, 1)
    predictions = {}
    for name in models:
        model = models[name]["model"]
        pred = model.predict(X_future)
        predictions[name] = np.maximum(0, pred).astype(int)
    return predictions


# ----- CHARTS -----

def chart_training_fit(years, accidents, models):
    set_dark_theme()
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)

    ax.plot(years, accidents / 1e6, "o-", color=WHITE, linewidth=2.5, markersize=6, label="Actual")
    ax.plot(years, models["linear"]["pred_all"] / 1e6, "--", color=BLUE, linewidth=1.8, alpha=0.85, label="Linear Regression")
    ax.plot(years, models["polynomial"]["pred_all"] / 1e6, "--", color=ORANGE, linewidth=1.8, alpha=0.85, label="Polynomial (deg 3)")
    ax.plot(years, models["random_forest"]["pred_all"] / 1e6, "--", color=GREEN, linewidth=1.8, alpha=0.85, label="Random Forest")

    ax.set_title("Model Training - Fitted vs Actual (2003-2015)", pad=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Accidents (Millions)")
    ax.set_xticks(years)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=9, facecolor=CARD, edgecolor="#242f45", labelcolor=WHITE)
    fig.tight_layout()
    return save_chart(fig, "chart_training.png")


def chart_train_test(years, accidents, models):
    set_dark_theme()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(BG)

    model_list = [
        ("linear",        "Linear Regression",  BLUE),
        ("polynomial",    "Polynomial (deg 3)", ORANGE),
        ("random_forest", "Random Forest",      GREEN),
    ]

    for ax, (key, title, color) in zip(axes, model_list):
        ax.set_facecolor(CARD)
        m = models[key]

        x_train = m["X_train"].ravel()
        x_test  = m["X_test"].ravel()
        y_train = m["y_train"] / 1e6
        y_test  = m["y_test"] / 1e6
        y_pred  = m["model"].predict(m["X_test"].reshape(-1, 1)) / 1e6

        ax.scatter(x_train, y_train, color=BLUE,  s=55, zorder=4, label="Train")
        ax.scatter(x_test,  y_test,  color=RED,   s=70, zorder=5, label="Test Actual")
        ax.scatter(x_test,  y_pred,  color=color, s=70, marker="D", zorder=6, label="Test Predicted")
        ax.plot(years, m["pred_all"] / 1e6, "--", color=color, alpha=0.5, linewidth=1.4)

        ax.set_title(f"{title}\nR2={m['r2_test']:.3f}  RMSE={m['rmse']/1e3:.1f}K", pad=10)
        ax.set_xlabel("Year")
        if ax == axes[0]:
            ax.set_ylabel("Accidents (M)")
        ax.legend(fontsize=8, facecolor=CARD, edgecolor="#242f45", labelcolor=WHITE)
        ax.grid(True, axis="y")

    fig.suptitle("Train / Test Split - All Three Models", color=WHITE, fontsize=14, y=1.01)
    fig.tight_layout()
    return save_chart(fig, "chart_train_test.png")


def chart_predictions(years, accidents, future_years, future_preds, models):
    set_dark_theme()
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)

    all_years = np.concatenate([years, future_years])

    ax.axvspan(2015.5, future_years[-1] + 0.5, color=ORANGE, alpha=0.04)
    ax.axvline(2015.5, color=GRAY, linestyle=":", linewidth=1)
    ax.text(2015.7, max(accidents) / 1e6 * 0.97, "PREDICTIONS ->", color=ORANGE, fontsize=8.5, fontweight="bold")

    ax.plot(years, accidents / 1e6, "o-", color=WHITE, linewidth=2.5, markersize=6, label="Historical Data", zorder=5)

    colors = {"linear": BLUE, "polynomial": ORANGE, "random_forest": GREEN}
    labels = {"linear": "Linear Reg", "polynomial": "Poly Reg (deg 3)", "random_forest": "Random Forest"}

    for key in colors:
        ax.plot(years, models[key]["pred_all"] / 1e6, "--", color=colors[key], linewidth=1.3, alpha=0.5)
        ax.plot(future_years, future_preds[key] / 1e6, "o-", color=colors[key], linewidth=2.2, markersize=7, label=labels[key])

    ax.set_title("Road Accident Predictions 2016-2025", pad=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Accidents (Millions)")
    ax.set_xticks(all_years)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=9, facecolor=CARD, edgecolor="#242f45", labelcolor=WHITE)
    fig.tight_layout()
    return save_chart(fig, "chart_predictions.png")


def chart_model_comparison(models):
    set_dark_theme()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor(BG)

    names  = ["Linear", "Polynomial", "Rnd Forest"]
    colors = [BLUE, ORANGE, GREEN]

    r2_vals   = [models["linear"]["r2_test"],  models["polynomial"]["r2_test"],  models["random_forest"]["r2_test"]]
    rmse_vals = [models["linear"]["rmse"]/1e3,  models["polynomial"]["rmse"]/1e3,  models["random_forest"]["rmse"]/1e3]
    mae_vals  = [models["linear"]["mae"]/1e3,   models["polynomial"]["mae"]/1e3,   models["random_forest"]["mae"]/1e3]

    for ax, (vals, title, ylabel) in zip(axes, [
        (r2_vals,   "R2 Score (Test Set)",    "R2"),
        (rmse_vals, "RMSE (Test, Thousands)", "RMSE x1000"),
        (mae_vals,  "MAE  (Test, Thousands)", "MAE  x1000"),
    ]):
        ax.set_facecolor(CARD)
        bars = ax.bar(names, vals, color=colors, width=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, color=WHITE, fontweight="bold")
        ax.set_title(title, pad=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")

    fig.suptitle("Model Performance Comparison", color=WHITE, fontsize=14, y=1.02)
    fig.tight_layout()
    return save_chart(fig, "chart_model_comparison.png")


def chart_yoy_change(years, accidents):
    set_dark_theme()
    yoy = np.diff(accidents) / accidents[:-1] * 100
    x   = years[1:]

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)

    bar_colors = [GREEN if v >= 0 else RED for v in yoy]
    ax.bar(x, yoy, color=bar_colors, width=0.6, zorder=3)
    ax.axhline(0, color=GRAY, linewidth=0.8)

    for xi, v in zip(x, yoy):
        offset = 0.1 if v >= 0 else -0.2
        align  = "bottom" if v >= 0 else "top"
        ax.text(xi, v + offset, f"{v:+.1f}%", ha="center", va=align, fontsize=8, color=WHITE)

    ax.set_title("Year-over-Year Change in National Road Accidents", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Change (%)")
    ax.set_xticks(x)
    ax.grid(True, axis="y")
    fig.tight_layout()
    return save_chart(fig, "chart_yoy.png")


def chart_top_states(df):
    set_dark_theme()
    year_cols = [c for c in df.columns if c.isdigit()]
    df["Total"] = df[year_cols].sum(axis=1)
    state_totals = df.groupby("State")["Total"].sum().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)

    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(state_totals)))
    bars = ax.barh(state_totals.index[::-1], state_totals.values[::-1] / 1e6,
                   color=colors[::-1], height=0.65, zorder=3)
    for bar, v in zip(bars, state_totals.values[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v/1e6:.2f}M", va="center", fontsize=9, color=WHITE)

    ax.set_title("Top 10 States - Total Accidents 2003-2015", pad=12)
    ax.set_xlabel("Total Accidents (Millions)")
    ax.grid(True, axis="x")
    fig.tight_layout()
    return save_chart(fig, "chart_top_states.png")


def chart_state_trends(df):
    set_dark_theme()
    year_cols = [c for c in df.columns if c.isdigit()]
    df["Total"] = df[year_cols].sum(axis=1)
    top5  = df.groupby("State")["Total"].sum().sort_values(ascending=False).head(5).index
    years = [int(y) for y in year_cols]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)

    colors = [RED, BLUE, ORANGE, GREEN, PURPLE]
    for state, color in zip(top5, colors):
        row = df[df["State"] == state][year_cols].sum()
        ax.plot(years, row.values / 1e3, "o-", color=color, linewidth=2, markersize=5, label=state)

    ax.set_title("Top 5 States - Year-Wise Accident Trend (2003-2015)", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Accidents (Thousands)")
    ax.set_xticks(years)
    ax.grid(True, axis="y")
    ax.legend(facecolor=CARD, edgecolor="#242f45", labelcolor=WHITE, fontsize=9)
    fig.tight_layout()
    return save_chart(fig, "chart_state_trends.png")


def chart_top_districts(df):
    set_dark_theme()
    year_cols = [c for c in df.columns if c.isdigit()]
    df["Total"] = df[year_cols].sum(axis=1)
    top15 = df.nlargest(15, "Total")[["District", "State", "Total"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)

    labels = [f"{r['District']}\n({r['State'][:8]})" for _, r in top15.iterrows()]
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(top15)))

    bars = ax.barh(labels[::-1], top15["Total"].values[::-1] / 1e6, color=colors, height=0.65, zorder=3)
    for bar, v in zip(bars, top15["Total"].values[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v/1e6:.2f}M", va="center", fontsize=8.5, color=WHITE)

    ax.set_title("Top 15 Districts - Total Accidents (2003-2015)", pad=12)
    ax.set_xlabel("Total Accidents (Millions)")
    ax.grid(True, axis="x")
    fig.tight_layout()
    return save_chart(fig, "chart_top_districts.png")


def chart_2015_pie(df):
    set_dark_theme()
    s2015 = df.groupby("State")["2015"].sum().sort_values(ascending=False)
    top8  = s2015.head(8)
    rest  = pd.Series({"Others": s2015[8:].sum()})
    data  = pd.concat([top8, rest])

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    colors = [RED, BLUE, ORANGE, GREEN, PURPLE, "#ff6b6b", "#ffd93d", "#6bcb77", GRAY]
    wedges, texts, autotexts = ax.pie(
        data.values, labels=data.index, autopct="%1.1f%%",
        colors=colors, startangle=140,
        pctdistance=0.78, labeldistance=1.08,
        wedgeprops={"linewidth": 1.2, "edgecolor": BG}
    )
    for t in texts:
        t.set_color(GRAY)
        t.set_fontsize(9)
    for a in autotexts:
        a.set_color(BG)
        a.set_fontsize(8)
        a.set_fontweight("bold")

    ax.set_title("State-wise Share of Accidents - 2015", pad=16, color=WHITE, fontsize=13)
    fig.tight_layout()
    return save_chart(fig, "chart_2015_pie.png")


def chart_heatmap(df):
    set_dark_theme()
    year_cols = [c for c in df.columns if c.isdigit()]
    corr = df[year_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)

    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax,
                cmap="RdYlGn", linewidths=0.4, linecolor=BG,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.set_title("Year-to-Year Correlation Heatmap", pad=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return save_chart(fig, "chart_heatmap.png")





def chart_outlier_boxplot(df):
    set_dark_theme()
    year_cols = [c for c in df.columns if c.isdigit()]
    df["Total"] = df[year_cols].sum(axis=1)

    # IQR outlier detection on district totals
    Q1  = df["Total"].quantile(0.25)
    Q3  = df["Total"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers    = df[df["Total"] > upper].sort_values("Total", ascending=False)
    non_outliers = df[df["Total"] <= upper]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(BG)

    # ── LEFT: boxplot of district totals by top 8 states ──────────────────
    ax = axes[0]
    ax.set_facecolor(CARD)

    top8_states = df.groupby("State")["Total"].sum().sort_values(ascending=False).head(8).index
    data_by_state = [df[df["State"] == s]["Total"].values / 1e3 for s in top8_states]
    colors_bp = [RED, BLUE, ORANGE, GREEN, PURPLE, "#ff6b6b", "#ffd93d", "#6bcb77"]

    bp = ax.boxplot(data_by_state, patch_artist=True, notch=False,
                    medianprops={"color": WHITE, "linewidth": 2.2},
                    whiskerprops={"color": GRAY, "linewidth": 1.2},
                    capprops={"color": GRAY, "linewidth": 1.2},
                    flierprops={"marker": "o", "markersize": 5,
                                "markerfacecolor": RED, "markeredgecolor": RED, "alpha": 0.7})
    for patch, col in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)

    ax.set_xticks(range(1, len(top8_states) + 1))
    ax.set_xticklabels([s[:13] for s in top8_states], rotation=20, ha="right", fontsize=8)
    ax.set_title("District Accident Distribution — Top 8 States (Boxplot)", pad=12)
    ax.set_ylabel("Total Accidents per District (Thousands)")
    ax.grid(True, axis="y")

    # ── RIGHT: horizontal scatter — normal vs outlier districts ──────────
    ax2 = axes[1]
    ax2.set_facecolor(CARD)

    ax2.scatter(non_outliers["Total"] / 1e3,
                range(len(non_outliers)),
                color=BLUE, s=18, alpha=0.4, label="Normal districts", zorder=3)
    ax2.scatter(outliers["Total"] / 1e3,
                range(len(non_outliers), len(non_outliers) + len(outliers)),
                color=RED, s=40, alpha=0.85, label=f"Outliers ({len(outliers)})", zorder=4)

    ax2.axvline(upper / 1e3, color=ORANGE, linewidth=1.5, linestyle="--",
                label=f"Upper fence = {upper/1e3:.1f}K")

    # label top 5 outlier districts
    for _, row in outliers.head(5).iterrows():
        idx = list(df.index).index(row.name)
        ax2.annotate(f"{row['District'][:14]}",
                     xy=(row["Total"] / 1e3, non_outliers.shape[0] + list(outliers.index).index(row.name)),
                     xytext=(8, 0), textcoords="offset points",
                     fontsize=7.5, color=WHITE, va="center")

    ax2.set_title(f"Outlier Detection — IQR Method  |  {len(outliers)} outlier districts found", pad=12)
    ax2.set_xlabel("Total Accidents 2003–2015 (Thousands)")
    ax2.set_ylabel("District Index")
    ax2.legend(facecolor=CARD, edgecolor="#242f45", labelcolor=WHITE, fontsize=9)
    ax2.grid(True, axis="x")

    fig.suptitle(
        f"Outlier Analysis  |  Q1={Q1/1e3:.1f}K  Q3={Q3/1e3:.1f}K  IQR={IQR/1e3:.1f}K  Fence={upper/1e3:.1f}K",
        color=WHITE, fontsize=12, y=1.01
    )
    fig.tight_layout()
    return save_chart(fig, "chart_outlier_boxplot.png")

# ----- STATISTICAL TESTS -----

def run_statistical_tests(df, years, accidents):
    year_cols = [c for c in df.columns if c.isdigit()]

    # 1. Shapiro-Wilk Normality Test
    # Tests if yearly national accident totals follow a normal distribution
    sw_stat, sw_p = shapiro(accidents)
    shapiro_result = {
        "test":        "Shapiro-Wilk Normality Test",
        "h0":          "Yearly accident data follows a normal distribution",
        "h1":          "Yearly accident data does NOT follow a normal distribution",
        "statistic":   round(float(sw_stat), 4),
        "p_value":     round(float(sw_p), 4),
        "alpha":       0.05,
        "reject_h0":   bool(sw_p < 0.05),
        "result":      "Reject H0 - Data is NOT normal" if sw_p < 0.05 else "Fail to Reject H0 - Data is Normal",
        "conclusion":  f"W = {sw_stat:.4f}, p = {sw_p:.4f}. At alpha=0.05, the yearly accident data {'does NOT follow' if sw_p < 0.05 else 'follows'} a normal distribution.",
    }

    # 2. One-Sample T-Test
    # Tests if the mean yearly accidents is significantly different from 500,000
    mu0 = 500000
    sample_mean = float(np.mean(accidents))
    sample_std  = float(np.std(accidents, ddof=1))
    t_stat, t_p = ttest_1samp(accidents, mu0)
    ttest_result = {
        "test":        "One-Sample T-Test",
        "h0":          f"Mean yearly accidents = {mu0:,}",
        "h1":          f"Mean yearly accidents != {mu0:,}",
        "test_mean":   mu0,
        "sample_mean": round(sample_mean, 2),
        "sample_std":  round(sample_std, 2),
        "statistic":   round(float(t_stat), 4),
        "p_value":     round(float(t_p), 4),
        "alpha":       0.05,
        "reject_h0":   bool(t_p < 0.05),
        "result":      "Reject H0 - Mean is significantly different" if t_p < 0.05 else "Fail to Reject H0 - Mean is NOT significantly different",
        "conclusion":  f"t = {t_stat:.4f}, p = {t_p:.4f}. Sample mean = {sample_mean:,.0f}. At alpha=0.05, the mean yearly accidents {'IS' if t_p < 0.05 else 'is NOT'} significantly different from {mu0:,}.",
    }

    # 3. Z-Test (One-Sample)
    # Same as T-test but uses Z distribution — suitable when population std is known
    # Here we use sample std as an approximation (n=13)
    z_stat = (sample_mean - mu0) / (sample_std / np.sqrt(len(accidents)))
    z_p    = float(2 * (1 - norm.cdf(abs(z_stat))))
    z_critical = 1.96  # alpha = 0.05 two-tailed
    ztest_result = {
        "test":        "One-Sample Z-Test",
        "h0":          f"Population mean yearly accidents = {mu0:,}",
        "h1":          f"Population mean yearly accidents != {mu0:,}",
        "test_mean":   mu0,
        "sample_mean": round(sample_mean, 2),
        "sample_std":  round(sample_std, 2),
        "statistic":   round(z_stat, 4),
        "p_value":     round(z_p, 4),
        "z_critical":  z_critical,
        "alpha":       0.05,
        "reject_h0":   bool(abs(z_stat) > z_critical),
        "result":      "Reject H0 - Significant difference from population mean" if abs(z_stat) > z_critical else "Fail to Reject H0 - No significant difference",
        "conclusion":  f"Z = {z_stat:.4f}, p = {z_p:.4f}. |Z| = {abs(z_stat):.4f} {'>' if abs(z_stat) > z_critical else '<='} Z_critical = {z_critical}. The mean yearly accidents {'IS' if abs(z_stat) > z_critical else 'is NOT'} significantly different from {mu0:,}.",
    }

    # 4. Chi-Square Goodness of Fit Test
    # Tests if accidents are uniformly distributed across North, South, East, West India
    state_totals = df.groupby("State")[year_cols].sum().sum(axis=1)

    north = ["Delhi", "Haryana", "Himachal Pradesh", "Jammu & Kashmir", "Punjab", "Rajasthan", "Uttar Pradesh", "Uttarakhand"]
    south = ["Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana"]
    east  = ["Bihar", "Jharkhand", "Odisha", "West Bengal", "Assam"]
    west  = ["Goa", "Gujarat", "Maharashtra", "Madhya Pradesh"]

    region_totals = {
        "North": float(sum(state_totals.get(s, 0) for s in north)),
        "South": float(sum(state_totals.get(s, 0) for s in south)),
        "East":  float(sum(state_totals.get(s, 0) for s in east)),
        "West":  float(sum(state_totals.get(s, 0) for s in west)),
    }
    observed     = np.array(list(region_totals.values()))
    expected_val = float(observed.sum() / 4)
    expected     = np.full(4, expected_val)
    chi2_stat, chi2_p = chisquare(observed, expected)

    chi2_result = {
        "test":          "Chi-Square Goodness of Fit Test",
        "h0":            "Accidents are uniformly distributed across all 4 regions",
        "h1":            "Accidents are NOT uniformly distributed - some regions have more",
        "observed":      {k: int(v) for k, v in region_totals.items()},
        "expected_each": round(expected_val, 2),
        "statistic":     round(float(chi2_stat), 4),
        "p_value":       round(float(chi2_p), 6) if chi2_p > 0.000001 else 0.0,
        "alpha":         0.05,
        "reject_h0":     bool(chi2_p < 0.05),
        "result":        "Reject H0 - Accidents are NOT uniformly distributed" if chi2_p < 0.05 else "Fail to Reject H0 - Uniform distribution",
        "conclusion":    f"Chi2 = {chi2_stat:.4f}, p = {chi2_p:.6f}. South India has the highest accidents ({int(region_totals['South']):,}). The distribution across regions is {'significantly unequal' if chi2_p < 0.05 else 'not significantly different from uniform'}.",
    }

    return {
        "shapiro":  shapiro_result,
        "ttest":    ttest_result,
        "ztest":    ztest_result,
        "chi2":     chi2_result,
    }

# ----- MAIN FUNCTION -----

def run_pipeline(data_path):
    df = load_data(data_path)
    years, accidents = get_national_totals(df)

    models = train_all_models(years, accidents)

    future_years = np.arange(2016, 2026)
    future_preds = predict_future_years(models, future_years)

    charts = {
        "training":      chart_training_fit(years, accidents, models),
        "train_test":    chart_train_test(years, accidents, models),
        "predictions":   chart_predictions(years, accidents, future_years, future_preds, models),
        "comparison":    chart_model_comparison(models),
        "yoy":           chart_yoy_change(years, accidents),
        "top_states":    chart_top_states(df.copy()),
        "state_trends":  chart_state_trends(df.copy()),
        "top_districts": chart_top_districts(df.copy()),
        "pie_2015":      chart_2015_pie(df.copy()),
        "heatmap":       chart_heatmap(df.copy()),
        "outlier_boxplot": chart_outlier_boxplot(df.copy()),
    }

    year_cols = [c for c in df.columns if c.isdigit()]
    df["Total"] = df[year_cols].sum(axis=1)
    peak_idx = int(np.argmax(accidents))

    kpis = {
        "total_accidents": int(accidents.sum()),
        "peak_year":        int(years[peak_idx]),
        "peak_val":         int(accidents[peak_idx]),
        "states_count":     int(df["State"].nunique()),
        "districts_count":  int(df["District"].nunique()),
        "avg_annual":       int(accidents.mean()),
    }

    metrics = {}
    for key in models:
        m = models[key]
        metrics[key] = {
            "r2_train": round(m["r2_train"], 4),
            "r2_test":  round(m["r2_test"],  4),
            "r2":       round(m["r2"],        4),
            "rmse":     round(m["rmse"],      2),
            "mae":      round(m["mae"],       2),
        }

    future_table = []
    for i, yr in enumerate(future_years):
        future_table.append({
            "year":          int(yr),
            "linear":        int(future_preds["linear"][i]),
            "polynomial":    int(future_preds["polynomial"][i]),
            "random_forest": int(future_preds["random_forest"][i]),
        })

    state_totals = df.groupby("State")["Total"].sum().sort_values(ascending=False)
    top_states = []
    for state, total in state_totals.head(10).items():
        top_states.append({
            "state": state,
            "total": int(total),
            "avg":   int(total / len(year_cols))
        })

    s2015 = df.groupby("State")["2015"].sum().sort_values(ascending=False).reset_index()
    s2015.columns = ["state", "accidents_2015"]
    max_val = int(s2015["accidents_2015"].max())

    state_ranking = []
    for i, row in s2015.iterrows():
        state_ranking.append({
            "rank":      i + 1,
            "state":     row["state"],
            "accidents": int(row["accidents_2015"]),
            "pct":       round(int(row["accidents_2015"]) / max_val * 100, 1)
        })

    stat_tests = run_statistical_tests(df.copy(), years, accidents)

    return {
        "charts":        charts,
        "kpis":          kpis,
        "metrics":       metrics,
        "future_table":  future_table,
        "top_states":    top_states,
        "state_ranking": state_ranking,
        "years":         years.tolist(),
        "accidents":     accidents.tolist(),
        "future_years":  future_years.tolist(),
        "future_preds":  {k: v.tolist() for k, v in future_preds.items()},
        "stat_tests":    stat_tests,
    }