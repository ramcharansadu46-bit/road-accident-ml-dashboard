import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import (
    shapiro, normaltest, kstest, levene, f_oneway,
    pearsonr, spearmanr, ttest_ind, ttest_1samp, chi2_contingency, mannwhitneyu
)
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# colors matching the existing dark dashboard
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
        "axes.titlesize":    12,
        "axes.labelsize":    10,
    })


def save_chart(fig, filename):
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"/static/charts/{filename}"


def interpret(p, alpha=0.05):
    if p < alpha:
        return f"Significant (p={p:.4f} < {alpha})"
    return f"Not significant (p={p:.4f} >= {alpha})"


# ----- DATA PREP -----

def prepare_data(df):
    year_cols = [c for c in df.columns if c.isdigit()]
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # fill NaN with 0 - Telangana rows have NaN for 2003-2013 because
    # the state was carved out of Andhra Pradesh only in 2014
    df[year_cols] = df[year_cols].fillna(0)

    # national yearly totals
    yearly = df[year_cols].sum()
    years = np.array([int(y) for y in year_cols])
    accidents = yearly.values.astype(float)

    # state totals
    state_totals = df.groupby("State")[year_cols].sum().sum(axis=1)
    # remove "All India" row if present (not in updated dataset, but kept as safety check)
    state_totals = state_totals[state_totals.index != "All India"]

    # district totals
    df["Total"] = df[year_cols].sum(axis=1)

    return years, accidents, state_totals, df, year_cols


# =====================================================================
# TEST 1 — Shapiro-Wilk Normality Test
# Does the yearly accident data follow a normal distribution?
# =====================================================================

def test_normality_shapiro(accidents):
    stat, p = shapiro(accidents)
    result = {
        "name":        "Shapiro-Wilk Normality Test",
        "h0":          "The yearly accident data follows a normal distribution",
        "h1":          "The data does NOT follow a normal distribution",
        "statistic":   round(float(stat), 4),
        "p_value":     round(float(p), 4),
        "result":      interpret(p),
        "conclusion":  "The accident data is normally distributed. Parametric tests are valid." if p >= 0.05
                       else "The accident data is NOT normally distributed. Non-parametric tests should be used.",
        "use_case":    "Checks whether yearly national accident totals follow a bell curve. Important before applying t-tests or ANOVA."
    }
    return result


def chart_normality(accidents, years):
    set_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    # histogram with normal curve
    ax = axes[0]
    ax.set_facecolor(CARD)
    ax.hist(accidents / 1e6, bins=6, color=BLUE, alpha=0.7, edgecolor=BG, zorder=3, density=True)
    mu, sigma = np.mean(accidents / 1e6), np.std(accidents / 1e6)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=RED, linewidth=2.5, label="Normal curve")
    ax.set_title("Histogram vs Normal Distribution", pad=10)
    ax.set_xlabel("Total Accidents (Millions)")
    ax.set_ylabel("Density")
    ax.legend(facecolor=CARD, labelcolor=WHITE, fontsize=9)
    ax.grid(True, axis="y")

    # Q-Q plot
    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    (osm, osr), (slope, intercept, r) = stats.probplot(accidents, dist="norm")
    ax2.scatter(osm, osr, color=BLUE, s=60, zorder=4, label="Data points")
    x_line = np.array([min(osm), max(osm)])
    ax2.plot(x_line, slope * x_line + intercept, color=RED, linewidth=2, label="Normal line")
    ax2.set_title("Q-Q Plot (Quantile-Quantile)", pad=10)
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    ax2.legend(facecolor=CARD, labelcolor=WHITE, fontsize=9)
    ax2.grid(True)

    stat, p = shapiro(accidents)
    fig.suptitle(f"Shapiro-Wilk Normality Test   |   W={stat:.4f}   p={p:.4f}", color=WHITE, fontsize=12, y=1.01)
    fig.tight_layout()
    return save_chart(fig, "stat_normality.png")


# =====================================================================
# TEST 2 — Augmented Dickey-Fuller (ADF) Stationarity Test
# Is the time series stationary (stable mean over time)?
# =====================================================================

def test_adf_stationarity(accidents):
    result_adf = adfuller(accidents, autolag="AIC")
    stat = result_adf[0]
    p    = result_adf[1]
    crit = result_adf[4]
    result = {
        "name":        "Augmented Dickey-Fuller (ADF) Stationarity Test",
        "h0":          "The time series has a unit root (non-stationary / has a trend)",
        "h1":          "The time series is stationary (no unit root)",
        "statistic":   round(float(stat), 4),
        "p_value":     round(float(p), 4),
        "critical_1":  round(float(crit["1%"]), 4),
        "critical_5":  round(float(crit["5%"]), 4),
        "critical_10": round(float(crit["10%"]), 4),
        "result":      interpret(p),
        "conclusion":  "The series is stationary — it has no long-term upward/downward drift." if p < 0.05
                       else "The series is NON-STATIONARY — it has a significant trend over time. Differencing may be needed for time-series forecasting.",
        "use_case":    "Checks whether the accident trend has a consistent upward/downward direction or fluctuates around a stable mean."
    }
    return result


def chart_adf(accidents, years):
    set_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    # original series
    ax = axes[0]
    ax.set_facecolor(CARD)
    ax.plot(years, accidents / 1e6, "o-", color=BLUE, linewidth=2.2, markersize=6)
    ax.axhline(np.mean(accidents / 1e6), color=RED, linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(accidents)/1e6:.2f}M")
    ax.fill_between(years, np.mean(accidents / 1e6) - np.std(accidents / 1e6),
                    np.mean(accidents / 1e6) + np.std(accidents / 1e6), color=BLUE, alpha=0.08)
    ax.set_title("Original Time Series (National Accidents)", pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Accidents (Millions)")
    ax.legend(facecolor=CARD, labelcolor=WHITE, fontsize=9)
    ax.grid(True, axis="y")

    # first difference
    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    diff = np.diff(accidents)
    ax2.plot(years[1:], diff / 1e3, "o-", color=GREEN, linewidth=2.2, markersize=6)
    ax2.axhline(0, color=RED, linestyle="--", linewidth=1.2)
    ax2.set_title("First Difference (Removes Trend)", pad=10)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Change (Thousands)")
    ax2.grid(True, axis="y")

    result_adf = adfuller(accidents, autolag="AIC")
    fig.suptitle(f"ADF Stationarity Test   |   ADF Stat={result_adf[0]:.4f}   p={result_adf[1]:.4f}", color=WHITE, fontsize=12, y=1.01)
    fig.tight_layout()
    return save_chart(fig, "stat_adf.png")


# =====================================================================
# TEST 3 — Pearson Correlation Test
# Is there a linear relationship between year number and accidents?
# =====================================================================

def test_pearson_correlation(years, accidents):
    r, p = pearsonr(years, accidents)
    result = {
        "name":       "Pearson Correlation Test",
        "h0":         "There is no linear correlation between year and accident count",
        "h1":         "There IS a significant linear correlation",
        "statistic":  round(float(r), 4),
        "p_value":    round(float(p), 4),
        "r_squared":  round(float(r**2), 4),
        "result":     interpret(p),
        "strength":   "Strong" if abs(r) > 0.7 else ("Moderate" if abs(r) > 0.4 else "Weak"),
        "direction":  "Positive (accidents increase over years)" if r > 0 else "Negative (accidents decrease over years)",
        "conclusion": f"Pearson r = {r:.4f}. {'Significant' if p < 0.05 else 'No significant'} linear relationship between year and accidents. R² = {r**2:.4f} meaning {r**2*100:.1f}% of variance in accidents is explained by year.",
        "use_case":   "Measures how strongly the year and the accident count move together in a straight line."
    }
    return result


def chart_pearson(years, accidents):
    set_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    # scatter with regression line
    ax = axes[0]
    ax.set_facecolor(CARD)
    ax.scatter(years, accidents / 1e6, color=BLUE, s=70, zorder=4, label="Data points")
    m, b = np.polyfit(years, accidents / 1e6, 1)
    x_line = np.array([years[0], years[-1]])
    ax.plot(x_line, m * x_line + b, color=RED, linewidth=2.2, label=f"Regression line (r={pearsonr(years, accidents)[0]:.3f})")
    ax.set_title("Scatter Plot — Year vs Accidents", pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Accidents (Millions)")
    ax.legend(facecolor=CARD, labelcolor=WHITE, fontsize=9)
    ax.grid(True)

    # residuals
    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    predicted = m * years + b
    residuals = (accidents / 1e6) - predicted
    ax2.bar(years, residuals, color=[GREEN if v >= 0 else RED for v in residuals], width=0.6, zorder=3)
    ax2.axhline(0, color=GRAY, linewidth=1)
    ax2.set_title("Residuals (Actual - Predicted)", pad=10)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Residual (Millions)")
    ax2.grid(True, axis="y")

    r, p = pearsonr(years, accidents)
    fig.suptitle(f"Pearson Correlation   |   r={r:.4f}   R²={r**2:.4f}   p={p:.4f}", color=WHITE, fontsize=12, y=1.01)
    fig.tight_layout()
    return save_chart(fig, "stat_pearson.png")


# =====================================================================
# TEST 4 — Spearman Rank Correlation
# Non-parametric version — does rank of year correlate with rank of accidents?
# =====================================================================

def test_spearman_correlation(years, accidents):
    r, p = spearmanr(years, accidents)
    result = {
        "name":       "Spearman Rank Correlation Test",
        "h0":         "There is no monotonic relationship between year and accidents",
        "h1":         "There IS a significant monotonic relationship",
        "statistic":  round(float(r), 4),
        "p_value":    round(float(p), 4),
        "result":     interpret(p),
        "strength":   "Strong" if abs(r) > 0.7 else ("Moderate" if abs(r) > 0.4 else "Weak"),
        "conclusion": f"Spearman rho = {r:.4f}. {'Significant' if p < 0.05 else 'No significant'} monotonic relationship. Unlike Pearson, this test does not assume normality and is more robust to outliers.",
        "use_case":   "Non-parametric alternative to Pearson. Used because Shapiro-Wilk shows the data may not be perfectly normal."
    }
    return result


def chart_spearman(years, accidents):
    set_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    # rank scatter
    ax = axes[0]
    ax.set_facecolor(CARD)
    year_ranks = stats.rankdata(years)
    acc_ranks  = stats.rankdata(accidents)
    ax.scatter(year_ranks, acc_ranks, color=PURPLE, s=70, zorder=4)
    for i, yr in enumerate(years):
        ax.annotate(str(yr), (year_ranks[i], acc_ranks[i]), fontsize=7.5,
                    color=GRAY, ha="left", va="bottom", xytext=(3, 3), textcoords="offset points")
    m2, b2 = np.polyfit(year_ranks, acc_ranks, 1)
    x_r = np.array([year_ranks[0], year_ranks[-1]])
    ax.plot(x_r, m2 * x_r + b2, color=RED, linewidth=2)
    ax.set_title("Spearman — Rank of Year vs Rank of Accidents", pad=10)
    ax.set_xlabel("Rank of Year")
    ax.set_ylabel("Rank of Accidents")
    ax.grid(True)

    # comparison bar: pearson vs spearman
    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    pr, _ = pearsonr(years, accidents)
    sr, _ = spearmanr(years, accidents)
    bars = ax2.bar(["Pearson r", "Spearman rho"], [abs(pr), abs(sr)],
                   color=[BLUE, PURPLE], width=0.45, zorder=3)
    for bar, v in zip(bars, [abs(pr), abs(sr)]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.4f}", ha="center", fontsize=12, color=WHITE, fontweight="bold")
    ax2.set_title("Pearson vs Spearman Correlation Strength", pad=10)
    ax2.set_ylabel("|Correlation Coefficient|")
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, axis="y")

    r, p = spearmanr(years, accidents)
    fig.suptitle(f"Spearman Rank Correlation   |   rho={r:.4f}   p={p:.4f}", color=WHITE, fontsize=12, y=1.01)
    fig.tight_layout()
    return save_chart(fig, "stat_spearman.png")


# =====================================================================
# TEST 5 — One-Sample T-Test
# Is the mean yearly accidents significantly different from 1,000,000?
# =====================================================================

def test_one_sample_ttest(accidents):
    population_mean = 1000000
    t_stat, p = ttest_1samp(accidents, population_mean)
    result = {
        "name":            "One-Sample T-Test",
        "h0":              f"The mean yearly accidents = {population_mean:,}",
        "h1":              f"The mean yearly accidents != {population_mean:,}",
        "test_value":      population_mean,
        "sample_mean":     round(float(np.mean(accidents)), 2),
        "statistic":       round(float(t_stat), 4),
        "p_value":         round(float(p), 4),
        "result":          interpret(p),
        "conclusion":      f"Sample mean = {np.mean(accidents):,.0f}. {'The mean is significantly different from 1,000,000.' if p < 0.05 else 'The mean is NOT significantly different from 1,000,000.'} India averages {'above' if np.mean(accidents) > population_mean else 'below'} 1 million accidents per year.",
        "use_case":        "Tests whether the average yearly accident count is statistically different from a benchmark value (1 million)."
    }
    return result


def chart_ttest(accidents, years):
    set_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    population_mean = 1000000

    # bar chart with mean line
    ax = axes[0]
    ax.set_facecolor(CARD)
    ax.bar(years, accidents / 1e6, color=BLUE, width=0.6, zorder=3, alpha=0.8)
    ax.axhline(population_mean / 1e6, color=RED, linewidth=2, linestyle="--", label=f"Benchmark = {population_mean/1e6:.1f}M")
    ax.axhline(np.mean(accidents) / 1e6, color=GREEN, linewidth=2, linestyle="-.", label=f"Sample mean = {np.mean(accidents)/1e6:.2f}M")
    ax.set_title("Yearly Accidents vs Benchmark (1M)", pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Accidents (Millions)")
    ax.legend(facecolor=CARD, labelcolor=WHITE, fontsize=9)
    ax.grid(True, axis="y")

    # t-distribution
    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    df_t = len(accidents) - 1
    x = np.linspace(-4, 4, 200)
    ax2.plot(x, stats.t.pdf(x, df_t), color=BLUE, linewidth=2.2, label="t-distribution")
    t_stat, _ = ttest_1samp(accidents, population_mean)
    ax2.axvline(t_stat, color=RED, linewidth=2, linestyle="--", label=f"t-stat = {t_stat:.3f}")
    # shade rejection regions
    crit_val = stats.t.ppf(0.975, df_t)
    x_left  = np.linspace(-4, -crit_val, 50)
    x_right = np.linspace(crit_val, 4, 50)
    ax2.fill_between(x_left,  stats.t.pdf(x_left, df_t),  color=RED, alpha=0.3, label="Rejection region (α=0.05)")
    ax2.fill_between(x_right, stats.t.pdf(x_right, df_t), color=RED, alpha=0.3)
    ax2.set_title("T-Distribution with Critical Region", pad=10)
    ax2.set_xlabel("t value")
    ax2.set_ylabel("Density")
    ax2.legend(facecolor=CARD, labelcolor=WHITE, fontsize=8)
    ax2.grid(True, axis="y")

    t_s, p = ttest_1samp(accidents, population_mean)
    fig.suptitle(f"One-Sample T-Test   |   t={t_s:.4f}   p={p:.4f}   Mean={np.mean(accidents)/1e6:.3f}M", color=WHITE, fontsize=12, y=1.01)
    fig.tight_layout()
    return save_chart(fig, "stat_ttest.png")


# =====================================================================
# TEST 6 — One-Way ANOVA
# Are accident means significantly different across top 5 states?
# =====================================================================

def test_anova(df, year_cols):
    state_totals = df.groupby("State")[year_cols].sum()
    state_totals = state_totals[state_totals.index != "All India"]
    top5 = state_totals.sum(axis=1).sort_values(ascending=False).head(5).index
    groups = [state_totals.loc[s].values for s in top5]
    f_stat, p = f_oneway(*groups)
    result = {
        "name":       "One-Way ANOVA",
        "h0":         "All top 5 states have the same mean annual accident count",
        "h1":         "At least one state has a significantly different mean",
        "groups":     list(top5),
        "statistic":  round(float(f_stat), 4),
        "p_value":    round(float(p), 6),
        "result":     interpret(p),
        "conclusion": f"F = {f_stat:.4f}. {'At least one state has a significantly different mean accident rate.' if p < 0.05 else 'No significant difference in mean accident rates across top 5 states.'}",
        "use_case":   "Compares whether the average annual accident counts differ significantly across the top 5 accident-prone states."
    }
    return result


def chart_anova(df, year_cols):
    set_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    state_totals = df.groupby("State")[year_cols].sum()
    state_totals = state_totals[state_totals.index != "All India"]
    top5 = state_totals.sum(axis=1).sort_values(ascending=False).head(5).index
    colors = [RED, BLUE, ORANGE, GREEN, PURPLE]

    # box plot
    ax = axes[0]
    ax.set_facecolor(CARD)
    data_for_box = [state_totals.loc[s].values / 1e3 for s in top5]
    bp = ax.boxplot(data_for_box, patch_artist=True, notch=False,
                    medianprops={"color": WHITE, "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color(GRAY)
    ax.set_xticks(range(1, len(top5) + 1))
    ax.set_xticklabels([s[:12] for s in top5], rotation=15, ha="right", fontsize=8)
    ax.set_title("Distribution of Annual Accidents — Top 5 States", pad=10)
    ax.set_ylabel("Accidents per Year (Thousands)")
    ax.grid(True, axis="y")

    # mean bar with error bars
    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    means = [np.mean(state_totals.loc[s].values / 1e3) for s in top5]
    stds  = [np.std(state_totals.loc[s].values / 1e3) for s in top5]
    x_pos = np.arange(len(top5))
    bars = ax2.bar(x_pos, means, color=colors, width=0.55, zorder=3, alpha=0.85)
    ax2.errorbar(x_pos, means, yerr=stds, fmt="none", color=WHITE, capsize=5, linewidth=1.5, zorder=5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s[:12] for s in top5], rotation=15, ha="right", fontsize=8)
    ax2.set_title("Mean Annual Accidents with Std Deviation", pad=10)
    ax2.set_ylabel("Mean Accidents (Thousands)")
    ax2.grid(True, axis="y")

    f_stat, p = f_oneway(*[state_totals.loc[s].values for s in top5])
    fig.suptitle(f"One-Way ANOVA   |   F={f_stat:.4f}   p={p:.6f}", color=WHITE, fontsize=12, y=1.01)
    fig.tight_layout()
    return save_chart(fig, "stat_anova.png")


# =====================================================================
# TEST 7 — Mann-Whitney U Test
# Non-parametric: compare first half (2003-2008) vs second half (2009-2015)
# =====================================================================

def test_mannwhitney(accidents):
    first_half  = accidents[:6]   # 2003-2008
    second_half = accidents[6:]   # 2009-2015
    u_stat, p = mannwhitneyu(first_half, second_half, alternative="two-sided")
    result = {
        "name":        "Mann-Whitney U Test",
        "h0":          "No difference in accident distributions between 2003-2008 and 2009-2015",
        "h1":          "Significant difference exists between the two periods",
        "group1":      "2003-2008",
        "group2":      "2009-2015",
        "mean1":       round(float(np.mean(first_half)), 2),
        "mean2":       round(float(np.mean(second_half)), 2),
        "statistic":   round(float(u_stat), 4),
        "p_value":     round(float(p), 4),
        "result":      interpret(p),
        "conclusion":  f"Mean 2003-2008: {np.mean(first_half):,.0f}  |  Mean 2009-2015: {np.mean(second_half):,.0f}. {'Significant difference between the two periods.' if p < 0.05 else 'No significant difference between the two periods.'}",
        "use_case":    "Non-parametric test comparing the first half of the dataset (pre-2009) vs the second half (post-2009). Does not assume normality."
    }
    return result


def chart_mannwhitney(accidents, years):
    set_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    first_half  = accidents[:6]
    second_half = accidents[6:]
    y1 = years[:6]
    y2 = years[6:]

    ax = axes[0]
    ax.set_facecolor(CARD)
    ax.bar(y1, first_half / 1e6, color=BLUE,   width=0.6, label="2003-2008", zorder=3)
    ax.bar(y2, second_half / 1e6, color=ORANGE, width=0.6, label="2009-2015", zorder=3)
    ax.axhline(np.mean(first_half) / 1e6, color=BLUE, linestyle="--", linewidth=1.5,
               label=f"Mean 2003-08 = {np.mean(first_half)/1e6:.2f}M")
    ax.axhline(np.mean(second_half) / 1e6, color=ORANGE, linestyle="--", linewidth=1.5,
               label=f"Mean 2009-15 = {np.mean(second_half)/1e6:.2f}M")
    ax.set_title("Period Comparison: 2003-2008 vs 2009-2015", pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Accidents (Millions)")
    ax.legend(facecolor=CARD, labelcolor=WHITE, fontsize=8)
    ax.grid(True, axis="y")

    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    data = [first_half / 1e6, second_half / 1e6]
    bp = ax2.boxplot(data, patch_artist=True, notch=False,
                     medianprops={"color": WHITE, "linewidth": 2})
    bp["boxes"][0].set_facecolor(BLUE)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(ORANGE)
    bp["boxes"][1].set_alpha(0.7)
    for el in ["whiskers", "caps"]:
        for item in bp[el]:
            item.set_color(GRAY)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["2003-2008", "2009-2015"])
    ax2.set_title("Box Plot Comparison (Two Periods)", pad=10)
    ax2.set_ylabel("Accidents (Millions)")
    ax2.grid(True, axis="y")

    u, p = mannwhitneyu(first_half, second_half, alternative="two-sided")
    fig.suptitle(f"Mann-Whitney U Test   |   U={u:.2f}   p={p:.4f}", color=WHITE, fontsize=12, y=1.01)
    fig.tight_layout()
    return save_chart(fig, "stat_mannwhitney.png")


# =====================================================================
# TEST 8 — Levene's Test for Equality of Variances
# Do top 5 states have equal spread/variance in their accident counts?
# =====================================================================

def test_levene(df, year_cols):
    state_totals = df.groupby("State")[year_cols].sum()
    state_totals = state_totals[state_totals.index != "All India"]
    top5 = state_totals.sum(axis=1).sort_values(ascending=False).head(5).index
    groups = [state_totals.loc[s].values for s in top5]
    w_stat, p = levene(*groups)
    variances = {s: round(float(np.var(state_totals.loc[s].values)), 2) for s in top5}
    result = {
        "name":       "Levene's Test for Equality of Variances",
        "h0":         "All top 5 states have equal variance in accident counts",
        "h1":         "At least one state has significantly different variance",
        "groups":     list(top5),
        "variances":  variances,
        "statistic":  round(float(w_stat), 4),
        "p_value":    round(float(p), 4),
        "result":     interpret(p),
        "conclusion": f"{'Variances are significantly different across states — heteroscedasticity detected.' if p < 0.05 else 'Variances are equal across states — homoscedasticity confirmed.'} This {'violates' if p < 0.05 else 'satisfies'} the ANOVA assumption of equal variances.",
        "use_case":   "Tests whether the spread of accident data is similar across the top 5 states. This is a prerequisite check for ANOVA."
    }
    return result


def chart_levene(df, year_cols):
    set_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    state_totals = df.groupby("State")[year_cols].sum()
    state_totals = state_totals[state_totals.index != "All India"]
    top5 = state_totals.sum(axis=1).sort_values(ascending=False).head(5).index
    colors = [RED, BLUE, ORANGE, GREEN, PURPLE]

    # variance bar chart
    ax = axes[0]
    ax.set_facecolor(CARD)
    variances = [np.std(state_totals.loc[s].values / 1e3) for s in top5]
    bars = ax.bar([s[:12] for s in top5], variances, color=colors, width=0.55, zorder=3)
    for bar, v in zip(bars, variances):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}K", ha="center", fontsize=9, color=WHITE, fontweight="bold")
    ax.set_title("Standard Deviation of Annual Accidents (Top 5 States)", pad=10)
    ax.set_ylabel("Std Dev (Thousands)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y")

    # violin / strip plot via seaborn
    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    plot_data = []
    for s, col in zip(top5, colors):
        vals = state_totals.loc[s].values / 1e3
        for v in vals:
            plot_data.append({"State": s[:12], "Accidents_K": v})
    plot_df = pd.DataFrame(plot_data)
    sns.stripplot(data=plot_df, x="State", y="Accidents_K", ax=ax2,
                  palette=dict(zip([s[:12] for s in top5], colors)),
                  size=8, jitter=True, zorder=4)
    ax2.set_title("Year-wise Accident Spread per State", pad=10)
    ax2.set_xlabel("")
    ax2.set_ylabel("Annual Accidents (Thousands)")
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(True, axis="y")

    w, p = levene(*[state_totals.loc[s].values for s in top5])
    fig.suptitle(f"Levene's Test   |   W={w:.4f}   p={p:.4f}", color=WHITE, fontsize=12, y=1.01)
    fig.tight_layout()
    return save_chart(fig, "stat_levene.png")


# =====================================================================
# TEST 9 — Linear Regression Significance (OLS)
# Is year a statistically significant predictor of accidents?
# =====================================================================

def test_ols_regression(years, accidents):
    X = add_constant(years.astype(float))
    y = accidents
    model = OLS(y, X).fit()
    slope  = model.params[1]
    intercept = model.params[0]
    p_year = model.pvalues[1]
    p_const = model.pvalues[0]
    r2 = model.rsquared
    f_stat = model.fvalue
    f_p = model.f_pvalue
    dw = durbin_watson(model.resid)
    result = {
        "name":        "OLS Linear Regression Significance Test",
        "h0":          "Year has no significant effect on accident count (slope = 0)",
        "h1":          "Year IS a significant predictor of accident count",
        "slope":       round(float(slope), 2),
        "intercept":   round(float(intercept), 2),
        "r_squared":   round(float(r2), 4),
        "f_statistic": round(float(f_stat), 4),
        "f_p_value":   round(float(f_p), 4),
        "p_year":      round(float(p_year), 4),
        "p_value":     round(float(p_year), 4),
        "durbin_watson": round(float(dw), 4),
        "result":      interpret(p_year),
        "conclusion":  f"Slope = {slope:,.0f} accidents/year. R² = {r2:.4f}. {'Year is a significant predictor.' if p_year < 0.05 else 'Year is NOT a significant predictor.'} Durbin-Watson = {dw:.4f} ({'no autocorrelation' if 1.5 < dw < 2.5 else 'possible autocorrelation in residuals'}).",
        "use_case":    "Full OLS regression test — checks whether year significantly predicts accident count, reports R², slope, F-statistic, and residual autocorrelation."
    }
    return result


def chart_ols(years, accidents):
    set_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    X = add_constant(years.astype(float))
    model = OLS(accidents, X).fit()
    predicted = model.predict(X)
    residuals = accidents - predicted

    ax = axes[0]
    ax.set_facecolor(CARD)
    ax.scatter(years, accidents / 1e6, color=BLUE, s=65, zorder=4, label="Actual")
    ax.plot(years, predicted / 1e6, color=RED, linewidth=2.5, label=f"OLS fit (R²={model.rsquared:.4f})")
    ax.fill_between(years, (predicted - np.std(residuals)) / 1e6,
                    (predicted + np.std(residuals)) / 1e6, color=RED, alpha=0.1, label="±1 Std")
    ax.set_title("OLS Regression Fit", pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Accidents (Millions)")
    ax.legend(facecolor=CARD, labelcolor=WHITE, fontsize=9)
    ax.grid(True)

    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    ax2.scatter(predicted / 1e6, residuals / 1e3, color=GREEN, s=60, zorder=4)
    ax2.axhline(0, color=RED, linewidth=1.5, linestyle="--")
    ax2.set_title("Residuals vs Fitted Values", pad=10)
    ax2.set_xlabel("Fitted Values (Millions)")
    ax2.set_ylabel("Residuals (Thousands)")
    ax2.grid(True)

    fig.suptitle(f"OLS Regression   |   Slope={model.params[1]:,.0f}/yr   R²={model.rsquared:.4f}   p={model.pvalues[1]:.4f}", color=WHITE, fontsize=12, y=1.01)
    fig.tight_layout()
    return save_chart(fig, "stat_ols.png")


# =====================================================================
# TEST 10 — Chi-Square Goodness of Fit
# Is the distribution of accidents across regions uniform?
# =====================================================================

def test_chi_square(df, year_cols):
    state_totals = df.groupby("State")[year_cols].sum().sum(axis=1)
    state_totals = state_totals[state_totals.index != "All India"]

    # group into 4 regions
    north  = ["Delhi", "Haryana", "Himachal Pradesh", "Jammu & Kashmir", "Punjab", "Rajasthan", "Uttar Pradesh", "Uttarakhand"]
    south  = ["Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana"]
    east   = ["Bihar", "Jharkhand", "Odisha", "West Bengal", "Assam"]
    west   = ["Goa", "Gujarat", "Maharashtra", "Madhya Pradesh"]

    region_totals = {
        "North": sum(state_totals.get(s, 0) for s in north),
        "South": sum(state_totals.get(s, 0) for s in south),
        "East":  sum(state_totals.get(s, 0) for s in east),
        "West":  sum(state_totals.get(s, 0) for s in west),
    }
    observed = np.array(list(region_totals.values()), dtype=float)
    expected = np.full(4, observed.sum() / 4)
    chi2, p = stats.chisquare(observed, expected)

    result = {
        "name":      "Chi-Square Goodness of Fit Test",
        "h0":        "Accidents are uniformly distributed across all 4 regions of India",
        "h1":        "Accidents are NOT uniformly distributed — some regions have significantly more",
        "observed":  {k: int(v) for k, v in region_totals.items()},
        "expected":  round(float(expected[0]), 2),
        "statistic": round(float(chi2), 4),
        "p_value":   round(float(p), 6),
        "result":    interpret(p),
        "conclusion": f"Chi² = {chi2:.4f}. {'Accidents are NOT uniformly distributed — regional inequality is statistically significant.' if p < 0.05 else 'No significant regional difference detected.'} South India has the highest concentration.",
        "use_case":  "Tests whether road accidents are equally spread across North, South, East, West India or concentrated in specific regions."
    }
    return result


def chart_chi_square(df, year_cols):
    set_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    state_totals = df.groupby("State")[year_cols].sum().sum(axis=1)
    state_totals = state_totals[state_totals.index != "All India"]
    north = ["Delhi","Haryana","Himachal Pradesh","Jammu & Kashmir","Punjab","Rajasthan","Uttar Pradesh","Uttarakhand"]
    south = ["Andhra Pradesh","Karnataka","Kerala","Tamil Nadu","Telangana"]
    east  = ["Bihar","Jharkhand","Odisha","West Bengal","Assam"]
    west  = ["Goa","Gujarat","Maharashtra","Madhya Pradesh"]

    region_totals = {
        "North": sum(state_totals.get(s, 0) for s in north),
        "South": sum(state_totals.get(s, 0) for s in south),
        "East":  sum(state_totals.get(s, 0) for s in east),
        "West":  sum(state_totals.get(s, 0) for s in west),
    }
    regions  = list(region_totals.keys())
    observed = list(region_totals.values())
    expected = sum(observed) / 4
    colors_r = [BLUE, RED, GREEN, ORANGE]

    ax = axes[0]
    ax.set_facecolor(CARD)
    x = np.arange(len(regions))
    width = 0.35
    b1 = ax.bar(x - width/2, [v / 1e6 for v in observed], width, color=colors_r, zorder=3, label="Observed")
    b2 = ax.bar(x + width/2, [expected / 1e6] * 4, width, color=GRAY, alpha=0.5, zorder=3, label="Expected (uniform)")
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_title("Observed vs Expected (Uniform) — By Region", pad=10)
    ax.set_ylabel("Total Accidents (Millions)")
    ax.legend(facecolor=CARD, labelcolor=WHITE, fontsize=9)
    ax.grid(True, axis="y")

    ax2 = axes[1]
    ax2.set_facecolor(BG)
    wedges, texts, autotexts = ax2.pie(
        observed, labels=regions, autopct="%1.1f%%",
        colors=colors_r, startangle=140,
        wedgeprops={"linewidth": 1.2, "edgecolor": BG},
        pctdistance=0.75
    )
    for t in texts: t.set_color(GRAY); t.set_fontsize(10)
    for a in autotexts: a.set_color(BG); a.set_fontsize(9); a.set_fontweight("bold")
    ax2.set_title("Regional Share of Total Accidents", pad=10)

    chi2, p = stats.chisquare(np.array(observed, dtype=float), np.full(4, expected))
    fig.suptitle(f"Chi-Square Test   |   χ²={chi2:.4f}   p={p:.6f}", color=WHITE, fontsize=12, y=1.01)
    fig.tight_layout()
    return save_chart(fig, "stat_chi_square.png")


# =====================================================================
# MAIN — run all 10 tests
# =====================================================================

def run_stats(data_path):
    df = pd.read_csv(data_path)
    years, accidents, state_totals, df, year_cols = prepare_data(df)

    print("  Running statistical tests...")

    tests = {
        "normality":   test_normality_shapiro(accidents),
        "adf":         test_adf_stationarity(accidents),
        "pearson":     test_pearson_correlation(years, accidents),
        "spearman":    test_spearman_correlation(years, accidents),
        "ttest":       test_one_sample_ttest(accidents),
        "anova":       test_anova(df, year_cols),
        "mannwhitney": test_mannwhitney(accidents),
        "levene":      test_levene(df, year_cols),
        "ols":         test_ols_regression(years, accidents),
        "chi_square":  test_chi_square(df, year_cols),
    }

    print("  Generating statistical charts...")

    charts = {
        "normality":   chart_normality(accidents, years),
        "adf":         chart_adf(accidents, years),
        "pearson":     chart_pearson(years, accidents),
        "spearman":    chart_spearman(years, accidents),
        "ttest":       chart_ttest(accidents, years),
        "anova":       chart_anova(df, year_cols),
        "mannwhitney": chart_mannwhitney(accidents, years),
        "levene":      chart_levene(df, year_cols),
        "ols":         chart_ols(years, accidents),
        "chi_square":  chart_chi_square(df, year_cols),
    }

    return {"tests": tests, "charts": charts}