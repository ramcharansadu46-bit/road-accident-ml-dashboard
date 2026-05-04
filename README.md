# 🚗 India Road Accident ML Dashboard
### District-Wise Analysis 2003–2015 | Predictions 2016–2025

---

## 📦 Project Structure

```
road_accident_ml/
├── app.py              ← Flask server (run this)
├── ml_engine.py        ← ML pipeline: data cleaning, training, prediction, charts
├── data.csv            ← Dataset (district-wise 2003–2015)
├── templates/
│   └── index.html      ← Dashboard HTML (dark theme)
└── static/
    └── charts/         ← Auto-generated matplotlib/seaborn charts
```

---

## 🔧 Installation

```bash
pip install flask scikit-learn numpy pandas matplotlib seaborn
```

---

## 🚀 Run the Dashboard

```bash
python app.py
```

Then open your browser at:  **http://127.0.0.1:5000**

---

## 🤖 ML Models (via scikit-learn)

| Model                  | Library               | Notes                        |
|------------------------|-----------------------|------------------------------|
| Linear Regression      | `sklearn.linear_model`| Baseline                     |
| Polynomial Regression  | `sklearn.pipeline` + `PolynomialFeatures(degree=3)` | Captures non-linear trend |
| Random Forest          | `sklearn.ensemble`    | 200 estimators, random_state=42 |

---

## 📊 Charts Generated (matplotlib + seaborn)

1. **Training Fit** — all models vs actual (2003–2015)
2. **Train/Test Split** — last 3 years held out as test
3. **Future Predictions** — 2016–2025 for all models
4. **Model Comparison** — R², RMSE, MAE bar charts
5. **Year-over-Year %** change in national accidents
6. **Top 10 States** — cumulative totals
7. **Top 5 State Trends** — year-wise line chart
8. **Top 15 Districts** — horizontal bar chart
9. **2015 Pie Chart** — state-wise share
10. **Correlation Heatmap** — seaborn, year-to-year

---

## 📐 Evaluation Metrics

- **R² Score** (train + test)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

---

## 🛠 Tech Stack

```
Python 3.x
├── pandas       — data loading, cleaning, aggregation
├── numpy        — numerical operations
├── scikit-learn — ML models, metrics, preprocessing
├── matplotlib   — all chart generation
├── seaborn      — heatmap
└── Flask        — local web server + HTML rendering
```
