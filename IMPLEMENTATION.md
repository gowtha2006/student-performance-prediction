# 🛠️ Student Academic Performance Prediction: Implementation Guide & Technical Architecture

This document provides a comprehensive, step-by-step breakdown of how the **Student Academic Performance Prediction System** was implemented from data science to deployment.

---

## 📑 Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Dataset Engineering & Preprocessing](#2-dataset-engineering--preprocessing)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Machine Learning Pipeline & Model Training](#4-machine-learning-pipeline--model-training)
5. [Evaluation Metrics & Comparative Analysis](#5-evaluation-metrics--comparative-analysis)
6. [Flask Web Backend Implementation](#6-flask-web-backend-implementation)
7. [Frontend Interface (Score Predictor)](#7-frontend-interface-score-predictor)
8. [Production Deployment Guide (Render.com)](#8-production-deployment-guide-rendercom)

---

## 1. System Architecture

```
+-------------------------------------------------------------------+
|                        1. DATASET LAYER                           |
|  - 1,200 Student Profiles (9 Academic, Behavioral & Social Inputs) |
|  - Target: Continuous Final Score (0 - 100)                       |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                  2. DATA PREPROCESSING & EDA                      |
|  - Zero missing values validation, outlier bounding               |
|  - 8 Correlation & Distribution visual charts saved in /static/   |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                3. MACHINE LEARNING ENGINE                         |
|  - 80/20 Train-Test Split (960 train / 240 unseen test samples)    |
|  - Models: Linear Regression, Decision Tree, Random Forest        |
|  - Model Persistence: .pkl artifacts & model_metrics.json         |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                   4. FLASK BACKEND SERVER                         |
|  - REST APIs: /api/predict, /api/metrics, /api/what-if            |
|  - Intelligent Decision Engine (Personalized Action Plan)         |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                 5. MODERN WEB USER INTERFACE                      |
|  - Dedicated Score Predictor with interactive sliders & presets   |
|  - Animated SVG Circular Gauge & Grade Badges (A+, B+, C, D/F)    |
|  - Radar Profile Chart vs. Class Cohort Averages                  |
+-------------------------------------------------------------------+
```

---

## 2. Dataset Engineering & Preprocessing

The dataset is located in `dataset/student_performance.csv` and was generated via `dataset/generate_dataset.py`.

### Feature Attributes:
| Attribute | Data Type | Range | Description |
| :--- | :--- | :--- | :--- |
| `study_hours` | Float | $1.0 - 10.0$ | Daily hours dedicated to self-study |
| `attendance` | Float | $45.0\% - 100.0\%$ | Percentage of lectures attended |
| `previous_score` | Float | $30.0 - 100.0$ | Score in prior semester/mid-term exam |
| `assignment_score` | Float | $30.0 - 100.0$ | Average assignment completion mark |
| `internal_marks` | Float | $8.0 - 25.0$ | Continuous internal assessment marks |
| `sleep_hours` | Float | $4.0 - 9.5$ | Average daily sleep duration |
| `participation` | Integer | $1 - 10$ | Active classroom engagement rating |
| `parental_support` | Integer | $1 - 3$ | Level of home guidance (1=Low, 2=Med, 3=High) |
| `extracurricular` | Integer | $0 / 1$ | Involvement in sports/clubs (0=No, 1=Yes) |
| **`final_score`** | **Float** | **$0.0 - 100.0$** | **Target Output (Predicted Final Score)** |

---

## 3. Exploratory Data Analysis (EDA)

During the data analysis phase (`notebooks/analysis.ipynb` and `train_model.py`), statistical patterns and feature relationships were visualized and exported to `static/img/eda/`:

1. **`score_distribution.png`**: Normal distribution centered at $\mu = 73.0$, $\sigma = 11.3$.
2. **`correlation_heatmap.png`**: Discovered that `previous_score` ($r = 0.81$), `study_hours` ($r = 0.58$), and `internal_marks` ($r = 0.52$) carry the highest correlation with final outcomes.
3. **`study_vs_score.png`**: Demonstrates a positive slope where every $+1.5\text{ hours/day}$ increases expected scores by approximately $4 - 6\text{ points}$.
4. **`attendance_vs_score.png`**: Confirms that students with $>85\%$ attendance are $4\times$ more likely to score above 80 points.
5. **`actual_vs_predicted.png`**: Scatter plot showing close alignment along the ideal $y=x$ regression line on unseen test data.

---

## 4. Machine Learning Pipeline & Model Training

The training workflow is implemented in `train_model.py`:

```python
# 1. Feature matrix and Target split
X = df[['study_hours', 'attendance', 'previous_score', 'assignment_score',
        'internal_marks', 'sleep_hours', 'participation', 'parental_support', 'extracurricular']]
y = df['final_score']

# 2. 80/20 Train-Test Partition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 3. Model Training
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
}
```

---

## 5. Evaluation Metrics & Comparative Analysis

All models were evaluated on the 20% test partition (240 unseen students) using standard regression metrics:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

### Model Performance Summary:
| Algorithm | MAE | MSE | RMSE | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **2.728** | **11.796** | **3.435** | **0.9124 (91.2%)** | 🏆 **Best Model** |
| **Random Forest Regressor** | **3.343** | **18.308** | **4.279** | **0.8641 (86.4%)** | **Strong Fit** |
| **Decision Tree Regressor** | **4.406** | **30.878** | **5.557** | **0.7708 (77.1%)** | **Baseline Fit** |

---

## 6. Flask Web Backend Implementation

The Flask application (`app.py`) provides:
- **Model Loader**: Loads `student_model.pkl` and `model_metrics.json` at startup.
- **`POST /api/predict`**: Accepts student features, runs model inference, bounds prediction between $[0, 100]$, classifies performance grade, and generates rule-based smart recommendations.
- **`GET /api/metrics`**: Serves model benchmark scores and dataset averages.

### Performance Classification Logic:
```python
def classify_performance(score):
    if score >= 85.0:
        return {'level': 'Excellent', 'grade': 'A+ (Distinction)', 'color': '#10b981'}
    elif score >= 70.0:
        return {'level': 'Good', 'grade': 'A / B+ (First Class)', 'color': '#38bdf8'}
    elif score >= 50.0:
        return {'level': 'Average', 'grade': 'B / C (Second Class)', 'color': '#f59e0b'}
    else:
        return {'level': 'Needs Improvement', 'grade': 'D / F (At-Risk)', 'color': '#f43f5e'}
```

---

## 7. Frontend Interface (Score Predictor)

- **`templates/index.html`**: Clean, accessible layout focusing solely on the Score Predictor.
- **`static/css/style.css`**: Modern dark glassmorphic design system using CSS variables, custom range sliders, glowing borders, and responsive grid layouts.
- **`static/js/app.js`**:
  - Live slider updates with auto-debounce inference.
  - Preset buttons (`Top Achiever`, `Average`, `At-Risk`).
  - Animated SVG circular gauge and number counter.
  - Chart.js Radar Chart comparing individual student vs. class cohort averages.

---

## 8. Production Deployment Guide (Render.com)

The project includes `Procfile` and `requirements.txt` ready for 1-click cloud deployment.

### Steps to Deploy on Render:
1. Log in to **[dashboard.render.com](https://dashboard.render.com)** using your GitHub account.
2. Click **New +** → **Web Service**.
3. Select `gowtha2006/student-performance-prediction`.
4. Configure:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click **Deploy Web Service** to get your public live URL.
