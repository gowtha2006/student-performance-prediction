# 🎓 Student Academic Performance Prediction
End-to-end Machine Learning project to predict student final examination scores and academic performance.  
**Assessment-1 | Machine Learning Project**

---

## 📋 Project Overview

| Component | Description | Marks |
| :--- | :--- | :---: |
| **1. Problem Identification** | Business/Academic problem, objectives, success metrics | 10 |
| **2. Dataset & Preprocessing** | Student Performance dataset, cleaning, feature engineering | 15 |
| **3. EDA & Visualization** | Statistical insights, correlation plots, feature relationships | 10 |
| **4. ML Algorithm Implementation** | Linear Regression, Decision Tree, Random Forest Regressors | 20 |
| **5. Model Evaluation** | MAE, MSE, RMSE, $R^2$ Score benchmark | 10 |
| **6. Model Improvement** | Hyperparameter tuning, Random Forest ensemble, Feature importance | 10 |
| **7. Application / UI** | Modern interactive Web App (Score Predictor + Smart Advisories) | 10 |
| **8. GitHub Repository** | Clean modular architecture, complete code, documentation | 5 |
| **9. Deployment** | Render.com cloud deployment & local run configuration | 5 |
| **10. Presentation & Viva** | Viva talking points, formulas, architectural explanation | 5 |
| **Total** | | **100** |

---

## 🎯 1. Problem Identification

### Academic Problem
Higher educational institutions often identify academically struggling students too late—usually after mid-term failures or final semester exams. Proactively predicting final scores based on early behavioral, attendance, and continuous assessment metrics allows educators to intervene with personalized tutoring and support before exams occur.

### Objective
Build a high-precision supervised **regression model** that predicts a student's final examination score ($y \in [0, 100]$) using academic history, attendance records, study hours, sleep habits, and behavioral attributes.

### Success Metrics
- **Primary Metric**: Coefficient of Determination ($R^2 \ge 0.85$)
- **Secondary Metrics**: Mean Absolute Error ($\text{MAE} \le 3.5\text{ pts}$), Root Mean Squared Error ($\text{RMSE} \le 4.5\text{ pts}$)
- **Target Achieved**: **$R^2 = 0.9124\text{ (91.24\%)}$** with **$\text{MAE} = 2.728\text{ pts}$**

### Stakeholders
- **Academic Counselors & Faculty**: Early identification of at-risk students for remedial sessions.
- **Students**: Real-time feedback on how increasing study hours or attendance will boost their grades.
- **Department Heads / Deans**: Cohort-level performance analytics across semester courses.

---

## 📊 2. Dataset & Preprocessing

- **Dataset**: Student Academic Performance Dataset (`dataset/student_performance.csv`)
- **Size**: 1,200 student records × 10 attributes

### Features
- **Demographics & Guidance**: `parental_support` (Low/Medium/High), `extracurricular` (0/1)
- **Academic History**: `previous_score` (30–100), `assignment_score` (30–100), `internal_marks` (8–25)
- **Study & Behavioral**: `study_hours` (1.0–10.0 hrs/day), `attendance` (45%–100%), `sleep_hours` (4.0–9.5 hrs), `participation` (1–10 scale)
- **Target Variable**: `final_score` (Continuous score 0–100)

### Preprocessing Steps
1. Validated dataset integrity: **0 missing values**, zero corrupted entries.
2. Handled outlier bounds for realistic academic scoring ranges ($0 \le \text{final\_score} \le 100$).
3. Verified absence of duplicate student profiles.
4. Stratified feature matrix and split into **80% Training (960 samples)** and **20% Testing (240 unseen samples)**.
5. Standardized feature scaling and validation via `train_model.py`.

---

## 🔍 3. EDA & Visualization

### Key Insights (saved in `static/img/eda/`):
- **Final Score Distribution**: Follows a normal Gaussian distribution centered at $\mu = 73.0$ and $\sigma = 11.3$.
- **Prior Exam Score Impact**: Strongest baseline predictor ($r = 0.81$) of semester success.
- **Study Hours Correlation**: Every additional $+1.5\text{ hrs/day}$ of dedicated study adds $+3.6\text{ to }+5.5\text{ points}$ to final scores ($r = 0.58$).
- **Attendance Threshold**: Students with $\ge 85\%$ attendance are $4\times$ more likely to secure First Class / Distinction.
- **Sleep Deprivation Effect**: Sleeping $<6.0\text{ hours/night}$ correlates with a noticeable drop in recall efficiency and internal marks.

### Visual Plots Generated:
1. `score_distribution.png` — Score histogram & Kernel Density Estimate (KDE)
2. `correlation_heatmap.png` — Collinearity matrix across all 10 variables
3. `study_vs_score.png` — Daily study hours vs final score regression slope
4. `attendance_vs_score.png` — Attendance percentage vs final score scatter
5. `previous_vs_score.png` — Previous exam score correlation plot
6. `feature_importance.png` — Random Forest Gini feature importance ranking
7. `model_comparison.png` — Multi-metric benchmark bar chart (MAE, RMSE, $R^2$)
8. `actual_vs_predicted.png` — Scatter plot along the ideal $y = x$ fit line

---

## 🤖 4. ML Algorithm Implementation

Three distinct supervised regression models were implemented and comparatively evaluated:

| Model | Why Chosen |
| :--- | :--- |
| **Linear Regression** | Interpretable parametric baseline modeling linear relationships between features |
| **Decision Tree Regressor** | Non-linear tree partitions capturing complex threshold boundaries (`max_depth=6`) |
| **Random Forest Regressor** | Powerful ensemble bagging of 150 randomized decision trees to reduce variance |

All models are trained via the scikit-learn pipeline in `train_model.py`.

---

## 📈 5. Model Evaluation

Evaluated on the **20% unseen test partition (240 students)**:

| Model | MAE | MSE | RMSE | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **2.728** | **11.796** | **3.435** | **0.9124** | 🏆 **Best Performer** |
| **Random Forest Regressor** | **3.343** | **18.308** | **4.279** | **0.8641** | **Strong Fit** |
| **Decision Tree Regressor** | **4.406** | **30.878** | **5.557** | **0.7708** | **Baseline Fit** |

### Mathematical Evaluation Formulas:
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

---

## 🚀 6. Model Improvement & Feature Importance

- **Ensemble Bagging**: Random Forest with 150 estimators and constrained max depth to prevent overfitting.
- **Feature Importance Ranking (Gini Importance)**:
  1. `previous_score`: **54.29%**
  2. `study_hours`: **24.54%**
  3. `internal_marks`: **8.85%**
  4. `assignment_score`: **5.71%**
  5. `attendance`: **3.30%**
  6. `sleep_hours`: **1.36%**
  7. `participation`: **1.35%**
  8. `parental_support`: **0.37%**
  9. `extracurricular`: **0.23%**
- **Model Persistence**: Exported as `model/student_model.pkl` and `model/model_metrics.json`.

---

## 💻 7. Application / UI

Dedicated Web Application interface (`templates/index.html`, `static/css/style.css`, `static/js/app.js`):
- **Interactive Profile Form**: Live range sliders with synchronized numeric badges.
- **Quick Presets**: One-click profile testing (`Top Achiever`, `Average`, `At-Risk`).
- **Real-Time Prediction Card**: Animated circular score progress gauge (`/100`), predicted Grade badge (`A+`, `A/B+`, `B/C`, `D/F`), and performance classification.
- **Radar Profile Chart**: Visualizes student attribute footprint against class cohort averages using Chart.js.
- **Personalized AI Action Plan**: Smart recommendations generated dynamically based on sleep deficits, low attendance, or assignment scores.

### Run Locally:
```powershell
pip install -r requirements.txt
python app.py
```
Visit: `http://127.0.0.1:5000`

---

## 📁 8. GitHub Repository Structure

```
student-performance-prediction/
│
├── dataset/
│   ├── generate_dataset.py       # Script generating 1,200 student dataset
│   └── student_performance.csv   # Comprehensive student dataset
│
├── model/
│   ├── student_model.pkl         # Best saved model artifact
│   ├── linear_model.pkl          # Linear Regression model artifact
│   ├── tree_model.pkl            # Decision Tree model artifact
│   ├── forest_model.pkl          # Random Forest model artifact
│   └── model_metrics.json        # Evaluation metrics & feature rankings
│
├── notebooks/
│   └── analysis.ipynb            # Complete EDA & training Jupyter Notebook
│
├── static/
│   ├── css/
│   │   └── style.css             # Dark glassmorphic design system
│   ├── js/
│   │   └── app.js                # Dynamic sliders & live prediction sync
│   └── img/eda/                  # 8 high-resolution EDA visualization charts
│       ├── score_distribution.png
│       ├── correlation_heatmap.png
│       ├── study_vs_score.png
│       ├── attendance_vs_score.png
│       ├── previous_vs_score.png
│       ├── feature_importance.png
│       ├── model_comparison.png
│       └── actual_vs_predicted.png
│
├── templates/
│   └── index.html                # Dedicated Score Predictor dashboard
│
├── app.py                        # Flask server & REST API endpoints
├── train_model.py                # ML pipeline training & evaluation script
├── Procfile                      # Render cloud deployment configuration
├── requirements.txt              # Library dependencies
├── IMPLEMENTATION.md             # Technical architecture document
└── README.md                     # Assessment project documentation
```

---

## 🌐 9. Deployment

### Option A – Render Cloud Deployment (Recommended)
1. Push repository to GitHub: `https://github.com/gowtha2006/student-performance-prediction`
2. Go to **[dashboard.render.com](https://dashboard.render.com)** → Click **New +** → **Web Service**.
3. Select this repository.
4. Set **Build Command**: `pip install -r requirements.txt` and **Start Command**: `gunicorn app:app`.
5. Click **Deploy** to receive a public URL (e.g. `https://student-performance-prediction.onrender.com`).

### Option B – Local Run
```powershell
python app.py
```
Open `http://localhost:5000` in any web browser.

---

## 🎤 10. Presentation & Viva

### Viva Talking Points:
1. **Why $R^2$ and RMSE over MAE alone?**
   - $R^2$ measures the proportion of variance explained by the model ($91.24\%$), while RMSE heavily penalizes large outlier errors, ensuring consistent predictions for students in all grade tiers.
2. **Feature Importance Interpretation**:
   - Previous academic performance ($54.3\%$) and daily study hours ($24.5\%$) represent nearly $80\%$ of the predictive power, confirming that consistent daily preparation strongly dictates exam outcomes.
3. **How the Model Handles Non-Linear Behavioral Dynamics**:
   - Discovered that sleep hours below 6.0 hours introduce exponential cognitive penalties, captured through non-linear feature interactions.
4. **Production Integration**:
   - The model can be integrated via REST APIs into university Learning Management Systems (Canvas / Moodle / Blackboard) to generate automated mid-term academic alerts.

---

## 🛠 Tech Stack
- **Language**: Python 3.12
- **Data Science & ML**: Scikit-Learn, Pandas, NumPy, Joblib
- **Data Visualization**: Matplotlib, Seaborn, Chart.js
- **Web Backend & Serving**: Flask, Gunicorn
- **Frontend UI**: Modern HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (ES6+), Lucide Icons

---

## 📌 How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/gowtha2006/student-performance-prediction.git
cd student-performance-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Retrain ML models & generate all EDA plots
python train_model.py

# 4. Launch the web application
python app.py
```

---

**Author**: `gowtha2006`  
**Date**: August 2026  
**Course / Assessment**: Assessment-1 – Machine Learning Project
