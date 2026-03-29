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

OUTPUTS:
<img width="1877" height="875" alt="image" src="https://github.com/user-attachments/assets/5ad92f58-16d3-49be-bc1a-311d782a5afc" />
<img width="1785" height="359" alt="image" src="https://github.com/user-attachments/assets/3ed49827-baa5-4c63-bf78-03c853e17dc0" />
<img width="1794" height="671" alt="image" src="https://github.com/user-attachments/assets/e6151552-8731-4e90-8753-e30389c24e3e" />
<img width="1664" height="264" alt="image" src="https://github.com/user-attachments/assets/d1d7300b-3275-4e28-ba4a-eac05b3fceb5" />
<img width="1674" height="737" alt="image" src="https://github.com/user-attachments/assets/d56fe4c4-45f2-4f7f-822f-c33dd2d2c030" />
<img width="977" height="1035" alt="image" src="https://github.com/user-attachments/assets/a0ce05c4-e62c-4eb8-968f-f568c066f883" />
<img width="951" height="1033" alt="image" src="https://github.com/user-attachments/assets/7e4ad3d1-fa2d-4bdf-8107-61290076cd66" />

THESE ARE THE SAMPLE OUTPUTS, THEY MAY VARY ACCORDING TO THE USER SELECTED CRIME QUERY PARAMETERS!!

OVERALL, IT WAS A VERY FANTASTIC EXPERIENCE IN BUILDING AN IDEA AND THEN TO COMPLETE IMPLEMENTATION, WHICH OFFERS AN INTERFACE WITH EXCELLENT STYLING!


DONE BY:
- Name: VAISHNAVI KOLAMUDI
- MCA GRADUATE🎓



