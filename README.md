MACHINE LEARNING BASED CRIME PATTERN ANALYSIS AND PREDICTION

A Machine Learning based Crime Pattern Analysis and Prediction system using Random Forest and XGBoost algorithms on Indian Crime Dataset with 40,160 records from 29 cities. Features an interactive Flask dashboard to analyze and predict crime domains based on city, age, gender and weapon.

---

TECHNOLOGIES USED?
- Python
- Flask (Web Framework)
- Random Forest (Crime Pattern Analysis)
- XGBoost (Crime Domain Prediction)
- HTML, CSS, JavaScript
- Chart.js (Data Visualization)
- Pandas, NumPy, Scikit-learn

---

DATASET NAME?
- Name: Crime Dataset India (crime_dataset_india.csv)
- Records: 40,160
- Cities: 29 Major Indian Cities
- Years: 2020 - 2024
- Features: City, Crime Code, Crime Description, Victim Age, Victim Gender, Weapon Used, Police Deployed, Crime Domain, Case Closed, Date of Occurrence

---

HOW TO RUN?
1. Install required libraries:
   pip install -r requirements.txt

2. Run the Flask application:
   python app.py

3. Open browser and go to:
   http://localhost:5000

---

CRIME QUERY PARAMETERS:
Select the following parameters to run analysis and prediction:
- City (29 Indian Cities)
- Victim Age (10 - 79)
- Victim Gender (Male, Female, Other)
- Weapon Used (Firearm, Knife, Blunt Object, Explosives, Poison, Other, None)
- Crime Domain (Fire Accident, Other Crime, Traffic Fatality, Violent Crime)

---

HOW IT WORKS?

RANDOM FOREST ANALYSIS (CLICK ANALYSE BUTTON):
When the user selects query parameters and clicks the Analyse button, the Random Forest model processes the input and provides the following outputs:
- Total City Crimes count for selected city
- Case Closed Rate percentage
- Average Police Deployed per case
- Total Domain Crimes count
- Matched Records count
- Detailed Crime Records Table showing Report Number, Date Reported, Time of Occurrence, Date of Occurrence, Crime Code, Crime Description, Police Deployed, Case Closed status, and Date Case Closed

XGBOOST PREDICTION (CLICK PREDICT BUTTON):
When the user selects query parameters and clicks the Predict button, the XGBoost model processes the input and provides the following outputs:
- Predicted Crime Domain (Fire Accident / Other Crime / Traffic Fatality / Violent Crime)
- Top predicted crimes list based on selected filters
- Yearly Crime Trend bar chart showing crime counts from 2020 to 2024
- Monthly Crime Trend line chart showing crime occurrence across all 12 months

---

MODEL ACCURACY:
- Random Forest Accuracy: 91.25%
- XGBoost Accuracy: 94.18%

---

PROJECT STRUCTURE:
project/
├── app.py
├── preprocessor.py
├── random_forest_model.py
├── xgboost_model.py
├── requirements.txt
├── data/
│   └── crime_dataset_india.csv
├── models/
│   ├── rf_model.pkl
│   └── xgb_model.pkl
├── templates/
│   └── index.html
└── static/
├── css/
│   └── style.css
└── js/
└── main.js


---

CRIME DOMAINS:
| Domain | Description |
|---|---|
| Violent Crime | Assault, Robbery, Homicide, Domestic Violence |
| Other Crime | Fraud, Burglary, Cybercrime, Drug Offense |
| Fire Accident | Arson, Firearm Offense |
| Traffic Fatality | Traffic Violation |

---

DONE BY:
- Name: VAISHNAVI KOLAMUDI
- MCA GRADUATE🎓



