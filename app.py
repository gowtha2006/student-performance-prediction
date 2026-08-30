import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load Trained Models & Metrics
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Load metadata
metrics_path = os.path.join(MODEL_DIR, 'model_metrics.json')
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        MODEL_METRICS = json.load(f)
else:
    MODEL_METRICS = {}

# Load ML models
MODELS = {}
try:
    MODELS['best'] = joblib.load(os.path.join(MODEL_DIR, 'student_model.pkl'))
    MODELS['linear'] = joblib.load(os.path.join(MODEL_DIR, 'linear_model.pkl'))
    MODELS['tree'] = joblib.load(os.path.join(MODEL_DIR, 'tree_model.pkl'))
    MODELS['forest'] = joblib.load(os.path.join(MODEL_DIR, 'forest_model.pkl'))
    print("Successfully loaded all machine learning models.")
except Exception as e:
    print(f"Warning: Could not load some models: {e}")

FEATURE_NAMES = [
    'study_hours', 'attendance', 'previous_score', 'assignment_score',
    'internal_marks', 'sleep_hours', 'participation', 'parental_support', 'extracurricular'
]

def classify_performance(score):
    if score >= 85.0:
        return {
            'level': 'Excellent',
            'badge_class': 'badge-excellent',
            'grade': 'A+ (Distinction)',
            'color': '#10b981',
            'summary': 'Outstanding academic performance with high probability of top rank.'
        }
    elif score >= 70.0:
        return {
            'level': 'Good',
            'badge_class': 'badge-good',
            'grade': 'A / B+ (First Class)',
            'color': '#38bdf8',
            'summary': 'Consistent, solid performance with strong understanding of core subjects.'
        }
    elif score >= 50.0:
        return {
            'level': 'Average',
            'badge_class': 'badge-average',
            'grade': 'B / C (Second Class)',
            'color': '#f59e0b',
            'summary': 'Moderate performance. Targeted revision needed to achieve top tier.'
        }
    else:
        return {
            'level': 'Needs Improvement',
            'badge_class': 'badge-warning',
            'grade': 'D / F (At-Risk)',
            'color': '#f43f5e',
            'summary': 'Academic performance is below expected benchmarks. Immediate intervention advised.'
        }

def generate_recommendations(features, score):
    recs = []
    
    # Attendance Analysis
    if features['attendance'] < 75.0:
        recs.append({
            'type': 'danger',
            'icon': 'alert-triangle',
            'title': 'Critical: Low Attendance',
            'text': f"Attendance is {features['attendance']:.1f}%, below the mandatory 75% threshold. Increasing attendance directly impacts internal marks and final scoring."
        })
    elif features['attendance'] < 85.0:
        recs.append({
            'type': 'info',
            'icon': 'calendar',
            'title': 'Attendance Boost',
            'text': 'Aim for 90%+ attendance to maximize internal marks and concept exposure.'
        })

    # Study Hours Analysis
    if features['study_hours'] < 3.0:
        recs.append({
            'type': 'warning',
            'icon': 'clock',
            'title': 'Increase Daily Study Time',
            'text': f"Currently studying {features['study_hours']:.1f} hrs/day. Increasing daily dedicated study by 1.5–2 hours can raise your score by an estimated 4–7 points."
        })
    elif features['study_hours'] >= 6.0:
        recs.append({
            'type': 'success',
            'icon': 'check-circle',
            'title': 'High Study Commitment',
            'text': 'Strong daily study routine maintained. Ensure active recall and practice testing rather than passive reading.'
        })

    # Sleep Hours
    if features['sleep_hours'] < 6.0:
        recs.append({
            'type': 'warning',
            'icon': 'moon',
            'title': 'Sleep Deficit Warning',
            'text': f"Sleeping only {features['sleep_hours']:.1f} hours/night harms memory consolidation and test recall. Target 7.0–8.0 hours of restful sleep."
        })
    elif features['sleep_hours'] > 9.0:
        recs.append({
            'type': 'info',
            'icon': 'sunrise',
            'title': 'Optimize Routine',
            'text': 'Ensure a structured daily routine to balance study, rest, and physical wellness.'
        })

    # Assignment & Previous Exam
    if features['assignment_score'] < 60.0:
        recs.append({
            'type': 'warning',
            'icon': 'file-text',
            'title': 'Assignment Consistency',
            'text': f"Assignment score is {features['assignment_score']:.1f}%. Completing assignments ahead of deadlines boosts conceptual mastery."
        })

    if features['previous_score'] < 55.0:
        recs.append({
            'type': 'danger',
            'icon': 'trending-up',
            'title': 'Foundational Revision',
            'text': 'Previous exam scores indicate conceptual gaps. Review previous exam question solutions and consult instructors during office hours.'
        })

    if features['internal_marks'] < 14.0:
        recs.append({
            'type': 'info',
            'icon': 'award',
            'title': 'Internal Assessments',
            'text': f"Internal marks ({features['internal_marks']:.1f}/25) can be elevated with quiz preparation and continuous lab submissions."
        })

    if features['participation'] < 5:
        recs.append({
            'type': 'info',
            'icon': 'message-circle',
            'title': 'Classroom Engagement',
            'text': 'Active participation in lectures and peer discussions improves understanding of complex topics.'
        })

    if not recs or score >= 85.0:
        recs.insert(0, {
            'type': 'success',
            'icon': 'award',
            'title': 'Excellent Momentum!',
            'text': 'Your current academic metrics, attendance, and study habits align with top percentile performers. Keep this disciplined routine!'
        })

    return recs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify(MODEL_METRICS)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Extract features with defaults if missing
        features_dict = {
            'study_hours': float(data.get('study_hours', 4.5)),
            'attendance': float(data.get('attendance', 85.0)),
            'previous_score': float(data.get('previous_score', 70.0)),
            'assignment_score': float(data.get('assignment_score', 65.0)),
            'internal_marks': float(data.get('internal_marks', 16.0)),
            'sleep_hours': float(data.get('sleep_hours', 7.0)),
            'participation': int(data.get('participation', 6)),
            'parental_support': int(data.get('parental_support', 2)),
            'extracurricular': int(data.get('extracurricular', 1))
        }

        # Select model
        model_type = data.get('model_type', 'best')
        model = MODELS.get(model_type, MODELS.get('best'))

        if model is None:
            return jsonify({'error': 'Model not loaded or trained'}), 500

        # Format input DataFrame
        input_df = pd.DataFrame([features_dict])
        
        # Predict
        raw_pred = model.predict(input_df)[0]
        final_score = round(float(np.clip(raw_pred, 0.0, 100.0)), 1)
        
        perf_info = classify_performance(final_score)
        recommendations = generate_recommendations(features_dict, final_score)
        
        # Compare with cohort averages
        cohort_avg = MODEL_METRICS.get('dataset_stats', {}).get('averages', {})
        comparison = {}
        for key in FEATURE_NAMES:
            user_val = features_dict[key]
            avg_val = cohort_avg.get(key, user_val)
            comparison[key] = {
                'user': user_val,
                'cohort_avg': avg_val,
                'diff': round(user_val - avg_val, 2)
            }

        return jsonify({
            'success': True,
            'predicted_score': final_score,
            'performance': perf_info,
            'recommendations': recommendations,
            'comparison': comparison,
            'model_used': model_type
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/what-if', methods=['POST'])
def what_if_simulation():
    try:
        data = request.get_json(force=True)
        base_features = data.get('base_features', {})
        adjustments = data.get('adjustments', {})

        model = MODELS.get('best', MODELS.get('linear'))
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Base prediction
        base_df = pd.DataFrame([base_features])
        base_score = round(float(np.clip(model.predict(base_df)[0], 0.0, 100.0)), 1)

        # Adjusted features
        simulated_features = base_features.copy()
        for key, delta in adjustments.items():
            if key in simulated_features:
                simulated_features[key] = max(0.0, simulated_features[key] + float(delta))

        sim_df = pd.DataFrame([simulated_features])
        sim_score = round(float(np.clip(model.predict(sim_df)[0], 0.0, 100.0)), 1)

        diff = round(sim_score - base_score, 1)

        return jsonify({
            'success': True,
            'base_score': base_score,
            'simulated_score': sim_score,
            'score_delta': diff,
            'base_perf': classify_performance(base_score),
            'simulated_perf': classify_performance(sim_score)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json(force=True)
        students = data.get('students', [])
        if not students:
            return jsonify({'error': 'No students provided'}), 400

        model = MODELS.get('best')
        df = pd.DataFrame(students)
        
        # Ensure all columns exist
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = MODEL_METRICS.get('dataset_stats', {}).get('averages', {}).get(col, 50.0)

        preds = model.predict(df[FEATURE_NAMES])
        preds_clipped = np.round(np.clip(preds, 0.0, 100.0), 1)

        results = []
        for i, p in enumerate(preds_clipped):
            results.append({
                'id': i + 1,
                'score': float(p),
                'performance': classify_performance(float(p))['level']
            })

        return jsonify({
            'success': True,
            'count': len(results),
            'predictions': results,
            'average_score': round(float(np.mean(preds_clipped)), 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Student Performance Predictor on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
