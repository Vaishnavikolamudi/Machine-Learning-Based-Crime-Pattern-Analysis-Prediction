import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'crime_dataset_india.csv')

CITIES = [
    'Agra', 'Ahmedabad', 'Bangalore', 'Bhopal', 'Chennai', 'Delhi',
    'Faridabad', 'Ghaziabad', 'Hyderabad', 'Indore', 'Jaipur', 'Kalyan',
    'Kanpur', 'Kolkata', 'Lucknow', 'Ludhiana', 'Meerut', 'Mumbai',
    'Nagpur', 'Nashik', 'Patna', 'Pune', 'Rajkot', 'Srinagar', 'Surat',
    'Thane', 'Varanasi', 'Vasai', 'Visakhapatnam'
]

GENDERS       = ['F', 'M', 'X']
GENDER_LABELS = {'F': 'Female', 'M': 'Male', 'X': 'Other'}
WEAPONS       = ['Blunt Object', 'Explosives', 'Firearm', 'Knife', 'None', 'Other', 'Poison']
DOMAINS       = ['Fire Accident', 'Other Crime', 'Traffic Fatality', 'Violent Crime']
DESCRIPTIONS  = [
    'ARSON', 'ASSAULT', 'BURGLARY', 'COUNTERFEITING', 'CYBERCRIME',
    'DOMESTIC VIOLENCE', 'DRUG OFFENSE', 'EXTORTION', 'FIREARM OFFENSE',
    'FRAUD', 'HOMICIDE', 'IDENTITY THEFT', 'ILLEGAL POSSESSION', 'KIDNAPPING',
    'PUBLIC INTOXICATION', 'ROBBERY', 'SEXUAL ASSAULT', 'SHOPLIFTING',
    'TRAFFIC VIOLATION', 'VANDALISM', 'VEHICLE - STOLEN'
]


AGE_RANGES = [
    (10, 17, '10-17 (Minor)'),
    (18, 25, '18-25 (Young Adult)'),
    (26, 35, '26-35 (Adult)'),
    (36, 50, '36-50 (Middle-aged)'),
    (51, 65, '51-65 (Senior Adult)'),
    (66, 79, '66-79 (Elderly)'),
]

AGE_BINS   = [10, 17, 25, 35, 50, 65, 79]
AGE_LABELS = [0, 1, 2, 3, 4, 5]

FEATURE_COLS = [
    'City_enc', 'Victim Age', 'AgeGroup', 'Victim Gender_enc',
    'Weapon Used_enc', 'Crime Code', 'Police Deployed',
    'Hour', 'Month', 'DayOfWeek', 'Year', 'ReportingDelay',
    'Crime Description_enc',
]


def load_and_preprocess():
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    
    df['Date Reported']      = pd.to_datetime(df['Date Reported'],      dayfirst=True, errors='coerce')
    df['Date of Occurrence'] = pd.to_datetime(df['Date of Occurrence'], dayfirst=True, errors='coerce')
    df['Time of Occurrence'] = pd.to_datetime(df['Time of Occurrence'], dayfirst=True, errors='coerce')
    df['Date Case Closed']   = pd.to_datetime(df['Date Case Closed'],   dayfirst=True, errors='coerce')

    
    df['Date of Occurrence'] = df['Date of Occurrence'].fillna(df['Date Reported'])

    df['Hour']           = df['Time of Occurrence'].dt.hour.fillna(0).astype(int)
    df['Month']          = df['Date of Occurrence'].dt.month.fillna(1).astype(int)
    df['DayOfWeek']      = df['Date of Occurrence'].dt.dayofweek.fillna(0).astype(int)
    df['Year']           = df['Date of Occurrence'].dt.year.fillna(2020).astype(int)
    df['ReportingDelay'] = (df['Date Reported'] - df['Date of Occurrence']).dt.days.fillna(0).astype(int)
    df['ReportingDelay'] = df['ReportingDelay'].clip(0, 365)

    for col in ['City', 'Victim Gender', 'Crime Domain', 'Crime Description', 'Case Closed']:
        df[col] = df[col].str.strip()


    df['Weapon Used'] = df['Weapon Used'].fillna('None').str.strip()

    df['CaseClosed_bin'] = (df['Case Closed'] == 'Yes').astype(int)

    encoders = {}
    for col in ['City', 'Victim Gender', 'Weapon Used', 'Crime Domain', 'Crime Description']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].fillna('Unknown'))
        encoders[col] = le

    df['AgeGroup'] = pd.cut(
        df['Victim Age'],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True
    ).astype(int)

    return df, encoders


def get_features_rf(df):
    return df[FEATURE_COLS].fillna(0), df['Crime Domain_enc']


def get_features_xgb(df):
    return df[FEATURE_COLS].fillna(0), df['Crime Domain_enc']


def encode_input(city, age, gender, weapon, domain, encoders,
                 crime_description=None, hour=12, month=6):
    age = max(10, min(79, int(age)))

    age_group = 0
    for i, (lo, hi, _) in enumerate(AGE_RANGES):
        if lo <= age <= hi:
            age_group = i
            break

    def safe(enc, val):
        return int(enc.transform([val])[0]) if val in enc.classes_ else 0

    return {
        'city_enc'   : safe(encoders['City'],             city),
        'age'        : age,
        'age_group'  : age_group,
        'gender_enc' : safe(encoders['Victim Gender'],    gender),
        'weapon_enc' : safe(encoders['Weapon Used'],      weapon),
        'domain_enc' : safe(encoders['Crime Domain'],     domain),
        'desc_enc'   : safe(encoders['Crime Description'], crime_description)
                       if crime_description else 0,
        'hour'       : hour,
        'month'      : month,
    }