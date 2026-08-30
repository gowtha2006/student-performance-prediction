import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style for high-quality dark theme charts
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = '#1e293b'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6

def setup_directories():
    os.makedirs('model', exist_ok=True)
    os.makedirs('static/img/eda', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)

def generate_eda_visualizations(df, output_dir='static/img/eda'):
    print("[1/5] Generating EDA Visualizations...")
    
    # 1. Final Score Distribution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    sns.histplot(df['final_score'], kde=True, color='#6366f1', bins=25, ax=ax, edgecolor='#4338ca', alpha=0.6)
    ax.axvline(df['final_score'].mean(), color='#10b981', linestyle='--', linewidth=2, label=f"Mean ({df['final_score'].mean():.1f})")
    ax.axvline(df['final_score'].median(), color='#f59e0b', linestyle=':', linewidth=2, label=f"Median ({df['final_score'].median():.1f})")
    ax.set_title('Final Score Distribution', fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
    ax.set_xlabel('Final Score (/100)', fontsize=11, color='#cbd5e1')
    ax.set_ylabel('Student Count', fontsize=11, color='#cbd5e1')
    ax.legend(facecolor='#0f172a', edgecolor='#334155')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'), facecolor='#0f172a')
    plt.close()

    # 2. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 260, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap='magma', annot=True, fmt='.2f', square=True,
                linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), facecolor='#0f172a')
    plt.close()

    # 3. Study Hours vs Final Score
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    sns.regplot(data=df, x='study_hours', y='final_score', ax=ax,
                scatter_kws={'alpha': 0.5, 'color': '#38bdf8', 's': 30},
                line_kws={'color': '#f43f5e', 'linewidth': 2.5})
    ax.set_title('Study Hours vs. Final Score', fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
    ax.set_xlabel('Daily Study Hours', fontsize=11, color='#cbd5e1')
    ax.set_ylabel('Final Score (/100)', fontsize=11, color='#cbd5e1')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'study_vs_score.png'), facecolor='#0f172a')
    plt.close()

    # 4. Attendance vs Final Score
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    sns.regplot(data=df, x='attendance', y='final_score', ax=ax,
                scatter_kws={'alpha': 0.5, 'color': '#a855f7', 's': 30},
                line_kws={'color': '#10b981', 'linewidth': 2.5})
    ax.set_title('Attendance (%) vs. Final Score', fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
    ax.set_xlabel('Attendance Percentage (%)', fontsize=11, color='#cbd5e1')
    ax.set_ylabel('Final Score (/100)', fontsize=11, color='#cbd5e1')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attendance_vs_score.png'), facecolor='#0f172a')
    plt.close()

    # 5. Previous Score vs Final Score
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    sns.regplot(data=df, x='previous_score', y='final_score', ax=ax,
                scatter_kws={'alpha': 0.5, 'color': '#fbbf24', 's': 30},
                line_kws={'color': '#06b6d4', 'linewidth': 2.5})
    ax.set_title('Previous Exam Score vs. Final Score', fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
    ax.set_xlabel('Previous Exam Score (/100)', fontsize=11, color='#cbd5e1')
    ax.set_ylabel('Final Score (/100)', fontsize=11, color='#cbd5e1')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'previous_vs_score.png'), facecolor='#0f172a')
    plt.close()

    print(" EDA Visualizations saved successfully.")

def train_and_evaluate():
    setup_directories()
    
    # Load dataset
    csv_path = os.path.join('dataset', 'student_performance.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run generate_dataset.py first.")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    generate_eda_visualizations(df)

    # Feature & Target Selection
    feature_cols = [
        'study_hours', 'attendance', 'previous_score', 'assignment_score',
        'internal_marks', 'sleep_hours', 'participation',
        'parental_support', 'extracurricular'
    ]
    target_col = 'final_score'

    X = df[feature_cols]
    y = df[target_col]

    # Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"\n[2/5] Training data: {X_train.shape[0]} samples, Testing data: {X_test.shape[0]} samples")

    # Define candidate models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    }

    metrics_summary = {}
    fitted_models = {}
    test_predictions = {}

    print("\n[3/5] Evaluating Machine Learning Models:")
    print("-" * 65)
    print(f"{'Model':<20} | {'MAE':<8} | {'MSE':<8} | {'RMSE':<8} | {'R² Score':<8}")
    print("-" * 65)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        metrics_summary[name] = {
            'mae': round(mae, 3),
            'mse': round(mse, 3),
            'rmse': round(rmse, 3),
            'r2': round(r2, 4)
        }
        fitted_models[name] = model
        test_predictions[name] = y_pred

        print(f"{name:<20} | {mae:<8.3f} | {mse:<8.3f} | {rmse:<8.3f} | {r2:<8.4f}")

    print("-" * 65)

    # Determine best model based on highest R2
    best_model_name = max(metrics_summary, key=lambda k: metrics_summary[k]['r2'])
    best_model = fitted_models[best_model_name]
    print(f"\n[BEST MODEL SELECTED]: {best_model_name} (R2 = {metrics_summary[best_model_name]['r2']})")

    # Save models
    joblib.dump(best_model, os.path.join('model', 'student_model.pkl'))
    joblib.dump(fitted_models['Linear Regression'], os.path.join('model', 'linear_model.pkl'))
    joblib.dump(fitted_models['Decision Tree'], os.path.join('model', 'tree_model.pkl'))
    joblib.dump(fitted_models['Random Forest'], os.path.join('model', 'forest_model.pkl'))

    # Extract Feature Importances (from Random Forest)
    rf_model = fitted_models['Random Forest']
    importances = rf_model.feature_importances_
    feature_importance_dict = {
        col: round(float(imp), 4) for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    }

    # Generate Model Comparison Chart
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    model_names = list(metrics_summary.keys())
    x = np.arange(len(model_names))
    width = 0.25

    mae_vals = [metrics_summary[m]['mae'] for m in model_names]
    rmse_vals = [metrics_summary[m]['rmse'] for m in model_names]
    r2_vals = [metrics_summary[m]['r2'] * 10 for m in model_names]  # scaled for visualization

    ax.bar(x - width, mae_vals, width, label='MAE', color='#38bdf8')
    ax.bar(x, rmse_vals, width, label='RMSE', color='#f43f5e')
    ax.bar(x + width, r2_vals, width, label='R² (x10)', color='#10b981')

    ax.set_title('Model Performance Benchmark Comparison', fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11, color='#cbd5e1')
    ax.legend(facecolor='#0f172a', edgecolor='#334155')
    ax.set_ylabel('Metric Value', fontsize=11, color='#cbd5e1')
    plt.tight_layout()
    plt.savefig('static/img/eda/model_comparison.png', facecolor='#0f172a')
    plt.close()

    # Generate Feature Importance Chart
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    feat_names = list(feature_importance_dict.keys())
    feat_vals = list(feature_importance_dict.values())
    y_pos = np.arange(len(feat_names))
    
    ax.barh(y_pos, feat_vals, align='center', color='#6366f1', edgecolor='#818cf8', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_', ' ').title() for f in feat_names], color='#cbd5e1')
    ax.invert_yaxis()
    ax.set_xlabel('Relative Importance (Gini Importance)', fontsize=11, color='#cbd5e1')
    ax.set_title('Feature Importance (Random Forest Regressor)', fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
    plt.tight_layout()
    plt.savefig('static/img/eda/feature_importance.png', facecolor='#0f172a')
    plt.close()

    # Generate Actual vs Predicted Plot for Best Model
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    best_y_pred = test_predictions[best_model_name]
    ax.scatter(y_test, best_y_pred, color='#38bdf8', alpha=0.6, s=35, edgecolors='none')
    min_val = min(y_test.min(), best_y_pred.min())
    max_val = max(y_test.max(), best_y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color='#f43f5e', linewidth=2, label='Perfect Fit ($y=x$)')
    ax.set_title(f'Actual vs Predicted Score ({best_model_name})', fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
    ax.set_xlabel('Actual Final Score (/100)', fontsize=11, color='#cbd5e1')
    ax.set_ylabel('Predicted Final Score (/100)', fontsize=11, color='#cbd5e1')
    ax.legend(facecolor='#0f172a', edgecolor='#334155')
    plt.tight_layout()
    plt.savefig('static/img/eda/actual_vs_predicted.png', facecolor='#0f172a')
    plt.close()

    # Compute Dataset Statistics & Cohort Averages for Frontend Reference
    dataset_stats = {
        'total_students': int(len(df)),
        'mean_final_score': round(float(df['final_score'].mean()), 2),
        'median_final_score': round(float(df['final_score'].median()), 2),
        'min_final_score': round(float(df['final_score'].min()), 2),
        'max_final_score': round(float(df['final_score'].max()), 2),
        'averages': {
            col: round(float(df[col].mean()), 2) for col in feature_cols
        }
    }

    # Save comprehensive metadata
    metadata = {
        'best_model': best_model_name,
        'features': feature_cols,
        'metrics': metrics_summary,
        'feature_importance': feature_importance_dict,
        'dataset_stats': dataset_stats
    }

    with open(os.path.join('model', 'model_metrics.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print("\n[5/5] All models, metrics, and charts generated and saved successfully!")
    return metadata

if __name__ == '__main__':
    train_and_evaluate()
