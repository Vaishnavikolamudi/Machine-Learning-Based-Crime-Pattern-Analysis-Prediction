import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from preprocessor import load_and_preprocess, get_features_rf, encode_input, FEATURE_COLS, CITIES, DOMAINS

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'rf_model.pkl')
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

_rf_model    = None
_rf_encoders = None
_rf_accuracy = None
_rf_report   = None
_df_cache    = None


def train_rf():
    global _rf_model, _rf_encoders, _rf_accuracy, _rf_report, _df_cache
    df, encoders = load_and_preprocess()
    _df_cache    = df
    _rf_encoders = encoders

    X, y = get_features_rf(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=2024, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=None,
        min_samples_split=2, min_samples_leaf=1,
        max_features='sqrt', bootstrap=True,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred  = rf.predict(X_test)
    acc_pct = round(accuracy_score(y_test, y_pred) * 100, 2)
    domain_names = encoders['Crime Domain'].classes_
    report = classification_report(
        y_test, y_pred, target_names=domain_names,
        output_dict=True, zero_division=0
    )
    _rf_model    = rf
    _rf_accuracy = acc_pct
    _rf_report   = report
    joblib.dump({'model': rf, 'encoders': encoders,
                 'accuracy': acc_pct, 'report': report}, MODEL_PATH)


def load_rf():
    global _rf_model, _rf_encoders, _rf_accuracy, _rf_report, _df_cache
    if _rf_model is not None:
        return
    if os.path.exists(MODEL_PATH):
        data         = joblib.load(MODEL_PATH)
        _rf_model    = data['model']
        _rf_encoders = data['encoders']
        _rf_report   = data['report']
        raw_acc      = data.get('accuracy', 0)
        _rf_accuracy = round(raw_acc * 100, 2) if raw_acc <= 1.0 else raw_acc
    else:
        train_rf()
    if _df_cache is None:
        df, _ = load_and_preprocess()
        _df_cache = df


def get_filtered(df, city, domain, gender, weapon, age=None):
    candidates = [
        df[(df['City']==city) & (df['Crime Domain']==domain) &
           (df['Victim Gender']==gender) & (df['Weapon Used']==weapon)],
        df[(df['City']==city) & (df['Crime Domain']==domain) &
           (df['Victim Gender']==gender)],
        df[(df['City']==city) & (df['Crime Domain']==domain)],
        df[df['City']==city],
    ]
    for f in candidates:
        if len(f) >= 5:
            return f
    return df[df['City']==city]


def compute_analysis_stats(df, city, domain, gender, weapon, age):
    city_df   = df[df['City'] == city]
    domain_df = df[df['Crime Domain'] == domain]
    filtered  = get_filtered(df, city, domain, gender, weapon, age)
    total     = len(filtered)

    return {
        'top_crimes_city'     : filtered['Crime Description'].value_counts().head(5).to_dict(),
        'case_closed_rate'    : round(filtered['CaseClosed_bin'].mean() * 100, 1) if total > 0 else 0,
        'avg_police_deployed' : round(filtered['Police Deployed'].mean(), 1) if total > 0 else 0,
        'total_city_crimes'   : len(city_df),
        'total_domain_crimes' : len(domain_df),
        'total_filtered'      : total,
        'weapon_distribution' : filtered['Weapon Used'].value_counts().to_dict(),
        'gender_distribution' : filtered['Victim Gender'].value_counts().to_dict(),
        'monthly_trend'       : filtered.groupby('Month').size().to_dict(),
    }


def fmt_date(val, f='%d-%m-%Y'):
    """Format date, return empty string if null"""
    try:
        if pd.notna(val):
            return val.strftime(f)
    except:
        pass
    return ''


def analyse(city, age, gender, weapon, domain, crime_description=None):
    load_rf()
    df  = _df_cache
    age = int(age)

    
    mask = (
        (df['City'] == city) &
        (df['Crime Domain'] == domain) &
        (df['Victim Gender'] == gender) &
        (df['Weapon Used'] == weapon) &
        (df['Victim Age'] >= max(10, age - 10)) &
        (df['Victim Age'] <= min(79, age + 10))
    )
    matched = df[mask]
    if len(matched) == 0:
        matched = df[(df['City']==city) & (df['Crime Domain']==domain) & (df['Victim Gender']==gender)]
    if len(matched) == 0:
        matched = df[(df['City']==city) & (df['Crime Domain']==domain)]
    if len(matched) == 0:
        matched = df[df['City']==city]
    matched = matched.head(20).copy()

    enc           = encode_input(city, age, gender, weapon, domain, _rf_encoders,
                                 crime_description=crime_description)
    domain_labels = _rf_encoders['Crime Domain'].classes_

    
    batch_rows = []
    for _, row in matched.iterrows():
        row_desc_enc = enc['desc_enc']
        if crime_description is None:
            cd = row.get('Crime Description', '')
            if cd in _rf_encoders['Crime Description'].classes_:
                row_desc_enc = int(_rf_encoders['Crime Description'].transform([cd])[0])
        batch_rows.append([
            enc['city_enc'], enc['age'], enc['age_group'],
            enc['gender_enc'], enc['weapon_enc'],
            row.get('Crime Code', 0), row.get('Police Deployed', 0),
            enc['hour'], enc['month'], 0, 2020, 0, row_desc_enc
        ])

    records = []
    if batch_rows:
        X_batch = pd.DataFrame(batch_rows, columns=FEATURE_COLS)
        preds   = _rf_model.predict(X_batch)
        probas  = _rf_model.predict_proba(X_batch)

        for i, (_, row) in enumerate(matched.iterrows()):
    
            date_occ = row.get('Date of Occurrence')
            if pd.isna(date_occ):
                date_occ = row.get('Date Reported')

            
            time_occ = row.get('Time of Occurrence')

        
            weapon_val = row.get('Weapon Used', '')
            if pd.isna(weapon_val) or weapon_val == '':
                weapon_val = 'Not Specified'

            records.append({
                'report_number'      : int(row['Report Number']),
                'date_reported'      : fmt_date(row.get('Date Reported'), '%d-%m-%Y %H:%M'),
                'time_of_occurrence' : fmt_date(time_occ, '%H:%M'),
                'date_of_occurrence' : fmt_date(date_occ, '%d-%m-%Y'),  
                'crime_code'         : int(row['Crime Code']),
                'crime_description'  : row['Crime Description'],
                'police_deployed'    : int(row['Police Deployed']),
                'case_closed'        : row['Case Closed'],
                'date_case_closed'   : fmt_date(row.get('Date Case Closed')),  
                'city'               : row['City'],
                'victim_age'         : int(row['Victim Age']),
                'victim_gender'      : row['Victim Gender'],
                'weapon_used'        : weapon_val,
                'crime_domain'       : row['Crime Domain'],
            })

    return {
        'records'        : records,
        'model_accuracy' : _rf_accuracy,
        'total_matched'  : len(records),
        'stats'          : compute_analysis_stats(df, city, domain, gender, weapon, age),
        'report'         : _rf_report,
    }