import os
import numpy as np
import pandas as pd

def generate_student_data(n_samples=1200, random_state=42):
    np.random.seed(random_state)
    
    # 1. Study hours per day (1 to 10 hours)
    study_hours = np.round(np.random.gamma(shape=3.5, scale=1.3, size=n_samples), 1)
    study_hours = np.clip(study_hours, 1.0, 10.0)
    
    # 2. Attendance percentage (45% to 100%)
    attendance = np.round(np.random.beta(a=7, b=2, size=n_samples) * 60 + 40, 1)
    attendance = np.clip(attendance, 45.0, 100.0)
    
    # 3. Previous exam score (30 to 100)
    previous_score = np.round(np.random.normal(loc=68, scale=14, size=n_samples), 1)
    previous_score = np.clip(previous_score, 30.0, 100.0)
    
    # 4. Assignment completion score (30 to 100)
    assignment_score = np.round(0.45 * previous_score + 0.35 * (attendance * 0.9) + np.random.normal(0, 8, size=n_samples), 1)
    assignment_score = np.clip(assignment_score, 30.0, 100.0)
    
    # 5. Internal marks (out of 25)
    internal_marks = np.round(0.18 * previous_score + 0.05 * study_hours + np.random.normal(3, 2, size=n_samples), 1)
    internal_marks = np.clip(internal_marks, 8.0, 25.0)
    
    # 6. Sleep hours per day (4 to 9.5 hours)
    sleep_hours = np.round(np.random.normal(loc=6.8, scale=1.1, size=n_samples), 1)
    sleep_hours = np.clip(sleep_hours, 4.0, 9.5)
    
    # 7. Class participation rating (1 to 10)
    participation = np.round(0.05 * attendance + 0.03 * previous_score + np.random.normal(0, 1.5, size=n_samples))
    participation = np.clip(participation, 1, 10).astype(int)
    
    # 8. Extracurricular activities (0 = No, 1 = Yes)
    extracurricular = np.random.choice([0, 1], size=n_samples, p=[0.42, 0.58])
    
    # 9. Parental Support level (1 = Low, 2 = Medium, 3 = High)
    parental_support = np.random.choice([1, 2, 3], size=n_samples, p=[0.25, 0.50, 0.25])
    
    # Target: Final Score (0 to 100)
    sleep_efficiency = np.where(sleep_hours < 6.0, (sleep_hours - 6.0) * 1.5, 0)
    
    raw_final_score = (
        0.28 * previous_score +
        0.22 * assignment_score +
        1.10 * internal_marks +
        2.40 * study_hours +
        0.18 * attendance +
        0.80 * participation +
        1.20 * parental_support +
        1.00 * extracurricular +
        sleep_efficiency +
        np.random.normal(0, 3.2, size=n_samples) - 10.0
    )
    
    final_score = np.round(np.clip(raw_final_score, 20.0, 100.0), 1)
    
    df = pd.DataFrame({
        'study_hours': study_hours,
        'attendance': attendance,
        'previous_score': previous_score,
        'assignment_score': assignment_score,
        'internal_marks': internal_marks,
        'sleep_hours': sleep_hours,
        'participation': participation,
        'parental_support': parental_support,
        'extracurricular': extracurricular,
        'final_score': final_score
    })
    
    os.makedirs('dataset', exist_ok=True)
    csv_path = os.path.join('dataset', 'student_performance.csv')
    df.to_csv(csv_path, index=False)
    print(f"Generated dataset with {len(df)} samples saved to: {csv_path}")
    print("\nDataset Summary Preview:")
    print(df.describe().round(2))
    return df

if __name__ == '__main__':
    generate_student_data()
