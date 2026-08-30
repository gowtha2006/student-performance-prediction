# 🎓 Student Academic Performance Prediction System (ML + Antigravity)

An end-to-end Machine Learning web application and analytics platform designed to predict students' final examination scores (0–100) and provide actionable academic advisories based on academic, behavioral, and personal factors.

---

## 🌟 Project Highlights

- **Machine Learning Algorithms**: Linear Regression (Parametric baseline), Decision Tree Regressor, and Random Forest Regressor (Ensemble 150 Estimators).
- **High Model Accuracy**: **$R^2 = 91.24\%$** and **$\text{MAE} = 2.73\text{ points}$** on unseen test data.
- **Interactive Web Application**: Flask REST API backend with modern dark glassmorphic UI, live dynamic sliders, animated radial gauge, real-time What-If intervention simulator, and Chart.js dashboards.
- **Actionable AI Advisories**: Evaluates risk factors (attendance deficit, sleep deprivation, assignment delays) and provides personalized recommendations.

---

## 📂 Project Architecture

```
student-performance-prediction/
│
├── dataset/
│   ├── generate_dataset.py       # Script to generate 1,200 sample student dataset
│   └── student_performance.csv   # Comprehensive student dataset
│
├── model/
│   ├── student_model.pkl         # Best saved model (Linear Regression / Random Forest)
│   ├── linear_model.pkl          # Linear Regression model artifact
│   ├── tree_model.pkl            # Decision Tree model artifact
│   ├── forest_model.pkl          # Random Forest model artifact
│   └── model_metrics.json        # Evaluation metrics, dataset stats, feature rankings
│
├── notebooks/
│   └── analysis.ipynb            # Jupyter Notebook with full EDA & training pipeline
│
├── static/
│   ├── css/
│   │   └── style.css             # Glassmorphic dark theme CSS design system
│   ├── js/
│   │   └── app.js                # Frontend dynamic logic, Chart.js integrations & API sync
│   └── img/eda/                  # High-resolution EDA visualization charts
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
│   └── index.html                # Responsive web application dashboard
│
├── app.py                        # Flask server & REST API endpoints
├── train_model.py                # ML Training, model comparison & evaluation script
├── requirements.txt              # Python library dependencies
└── README.md                     # Academic documentation & evaluation guide
```

---

## 📊 College Project Evaluation Rubric (100 Marks Breakdown)

### 1. Problem Identification (10 Marks)
- **Objective**: Academic institutions require early-warning predictive systems to identify struggling students prior to final semester examinations.
- **ML Formulation**: Regression problem mapping 9 multidimensional attributes $\mathbf{X}$ to a continuous target score $y \in [0, 100]$.

### 2. Dataset & Preprocessing (15 Marks)
- **Total Records**: 1,200 student academic profiles.
- **Feature Matrix**:
  1. `study_hours`: Dedicated daily study hours (1.0 to 10.0 hrs).
  2. `attendance`: Lecture and practical attendance percentage (45% to 100%).
  3. `previous_score`: Prior semester/mid-term exam score (30 to 100).
  4. `assignment_score`: Homework and assignment completion rating (30 to 100).
  5. `internal_marks`: Continuous assessment marks (8 to 25 max).
  6. `sleep_hours`: Daily average sleep duration (4.0 to 9.5 hrs).
  7. `participation`: Classroom active engagement rating (1 to 10 scale).
  8. `parental_support`: Home guidance environment (1 = Low, 2 = Medium, 3 = High).
  9. `extracurricular`: Sports / club participation (0 = No, 1 = Yes).
- **Target Variable**: `final_score` (Continuous score 0–100).
- **Data Preprocessing**: Verification of zero missing values, duplicate removal, outlier bounding, and 80/20 train-test partition.

### 3. Exploratory Data Analysis & Visualization (15 Marks)
- **Score Distribution**: Gaussian distribution with $\mu = 73.0$ and $\sigma = 11.3$.
- **Correlation Matrix**: Uncovers that `previous_score` ($r = 0.81$), `study_hours` ($r = 0.58$), and `internal_marks` ($r = 0.52$) have the strongest positive correlation with final grades.
- **Non-Linear Dynamics**: Discovered that sleep hours below 6.0 hours introduce negative cognitive penalties.

### 4. Model Building (20 Marks)
Implemented three distinct regression algorithms for comparative benchmark analysis:
1. **Linear Regression**: Ordinary Least Squares baseline fitting linear weights.
2. **Decision Tree Regressor**: Non-linear tree partition (`max_depth = 6`).
3. **Random Forest Regressor**: Ensemble bagging of 150 decision trees (`max_depth = 10`, `n_jobs = -1`).

### 5. Model Evaluation (15 Marks)

Evaluated on 20% unseen test partition (240 students):

| Model | MAE | MSE | RMSE | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **2.728** | **11.796** | **3.435** | **0.9124 (91.2%)** | 🏆 **Best Performer** |
| **Random Forest Regressor** | **3.343** | **18.308** | **4.279** | **0.8641 (86.4%)** | **Strong Fit** |
| **Decision Tree Regressor** | **4.406** | **30.878** | **5.557** | **0.7708 (77.1%)** | **Baseline Fit** |

#### Mathematical Formulas
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

### 6. Application Development (15 Marks)
- **Interactive UI**: Clean dark glassmorphic interface with reactive sliders, circular score progress animations, and preset buttons.
- **Radar Profile Chart**: Visualizes student attribute footprint against class cohort averages.
- **What-If Intervention Simulator**: Real-time slider simulator calculating expected score increases given study and attendance improvements.
- **Classroom Batch Predictor**: Simulates batch predictions for whole student cohorts with grade distribution charts.

### 7. Conclusion & Practical Utility (10 Marks)
- Enables professors, counselors, and students to proactively forecast academic trajectories.
- Demonstrates how targeted interventions (such as adding 1.5 daily study hours or improving attendance above 85%) directly shift a student from "Average" to "Distinction".

---

## 🚀 How to Run the Project Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-run ML Pipeline & Chart Generation
```bash
python train_model.py
```

### 3. Launch Flask Web Application
```bash
python app.py
```

Open your browser and visit: `http://127.0.0.1:5000`

---

## 👨‍💻 Technologies Used
- **Backend**: Python 3.12, Flask
- **Machine Learning**: Scikit-Learn, NumPy, Pandas, Joblib
- **Data Visualization**: Matplotlib, Seaborn, Chart.js
- **Frontend UI**: Modern HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (ES6+), Lucide Icons
