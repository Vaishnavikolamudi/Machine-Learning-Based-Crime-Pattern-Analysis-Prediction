import numpy as np
import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from preprocessor import load_and_preprocess, get_features_xgb, encode_input, FEATURE_COLS, DOMAINS, DESCRIPTIONS, CITIES

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'xgb_model.pkl')
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

_xgb_model    = None
_xgb_encoders = None
_xgb_accuracy = None
_xgb_report   = None
_df_cache     = None

ALL_YEARS = [2020, 2021, 2022, 2023, 2024]


def train_xgb():
    global _xgb_model, _xgb_encoders, _xgb_accuracy, _xgb_report, _df_cache
    df, encoders  = load_and_preprocess()
    _df_cache     = df
    _xgb_encoders = encoders

    X, y = get_features_xgb(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=10, stratify=y
    )
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6,
        learning_rate=0.1, subsample=0.85,
        colsample_bytree=0.8, min_child_weight=1,
        gamma=0.02, reg_alpha=0.05, reg_lambda=1.0,
        eval_metric='mlogloss', early_stopping_rounds=30,
        random_state=42, n_jobs=-1, tree_method='hist'
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)

    y_pred  = xgb.predict(X_test)
    acc_pct = round(accuracy_score(y_test, y_pred) * 100, 2)
    domain_names = encoders['Crime Domain'].classes_
    report = classification_report(
        y_test, y_pred, target_names=domain_names,
        output_dict=True, zero_division=0
    )
    _xgb_model    = xgb
    _xgb_accuracy = acc_pct
    _xgb_report   = report
    joblib.dump({'model': xgb, 'encoders': encoders,
                 'accuracy': acc_pct, 'report': report}, MODEL_PATH)


def load_xgb():
    global _xgb_model, _xgb_encoders, _xgb_accuracy, _xgb_report, _df_cache
    if _xgb_model is not None:
        return
    if os.path.exists(MODEL_PATH):
        data          = joblib.load(MODEL_PATH)
        _xgb_model    = data['model']
        _xgb_encoders = data['encoders']
        _xgb_report   = data['report']
        raw_acc       = data.get('accuracy', 0)
        _xgb_accuracy = round(raw_acc * 100, 2) if raw_acc <= 1.0 else raw_acc
    else:
        train_xgb()
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


def predict(city, age, gender, weapon, domain, crime_description=None):
    load_xgb()
    df       = _df_cache
    encoders = _xgb_encoders
    age      = int(age)

    enc      = encode_input(city, age, gender, weapon, domain, encoders,
                            crime_description=crime_description)
    filtered = get_filtered(df, city, domain, gender, weapon, age)
    total    = len(filtered)
    avg_police = int(filtered['Police Deployed'].mean()) if total > 0 else 10

    avg_crime_code = 300
    if crime_description:
        desc_rows = df[df['Crime Description'] == crime_description]
        if len(desc_rows) > 0:
            avg_crime_code = int(desc_rows['Crime Code'].mode()[0])

    inputs = pd.DataFrame([
        [enc['city_enc'], enc['age'], enc['age_group'], enc['gender_enc'],
         enc['weapon_enc'], avg_crime_code, avg_police,
         hour, 6, 0, 2020, 1, enc['desc_enc']]
        for hour in [8, 14, 20]
    ], columns=FEATURE_COLS)

    domain_proba  = _xgb_model.predict_proba(inputs)
    avg_proba     = domain_proba.mean(axis=0)
    domain_labels = encoders['Crime Domain'].classes_

    domain_results = sorted(
        [{'domain': dom, 'probability': round(float(avg_proba[i]) * 100, 2)}
         for i, dom in enumerate(domain_labels)],
        key=lambda x: x['probability'], reverse=True
    )

    predicted_domain = domain_results[0]['domain']
    predicted_prob   = domain_results[0]['probability']

    
    top_crimes = filtered['Crime Description'].value_counts().head(5)
    crime_list = []
    for crime, count in top_crimes.items():
        pct = round(count / total * 100, 1) if total > 0 else 0
        crime_list.append({
            'crime'      : crime,
            'count'      : int(count),
            'percentage' : pct,
            'risk_level' : 'High' if pct > 20 else ('Medium' if pct > 10 else 'Low'),
        })

    
    raw_yearly = filtered.groupby('Year').size().to_dict()
    
    yearly_counts = {str(yr): int(raw_yearly.get(yr, 0)) for yr in ALL_YEARS}

    
    raw_monthly = filtered.groupby('Month').size().to_dict()
    monthly_counts = {str(m): int(raw_monthly.get(m, 0)) for m in range(1, 13)}

    
    crime_summary = {
        'city'            : city,
        'domain'          : domain,
        'gender'          : gender,
        'weapon'          : weapon,
        'total_records'   : total,
        'case_closed_rate': round(filtered['CaseClosed_bin'].mean() * 100, 1) if total > 0 else 0,
        'avg_police'      : round(filtered['Police Deployed'].mean(), 1) if total > 0 else 0,
        'top_crimes'      : crime_list,
        'predicted_domain': predicted_domain,
        'predicted_prob'  : predicted_prob,
    }

    return {
        'domain_probabilities': domain_results,
        'predicted_domain'    : predicted_domain,
        'predicted_probability': predicted_prob,
        'model_accuracy'      : _xgb_accuracy,
        'report'              : _xgb_report,
        'crime_summary'       : crime_summary,
        'yearly_counts'       : yearly_counts,    
        'monthly_counts'      : monthly_counts,   
    }