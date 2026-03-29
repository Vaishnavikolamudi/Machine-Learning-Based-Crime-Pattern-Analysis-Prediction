import sys
import os
import math
import logging
from flask import Flask, render_template, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

from preprocessor import (CITIES, GENDERS, GENDER_LABELS, WEAPONS, DOMAINS, DESCRIPTIONS, AGE_RANGES)
from random_forest_model import analyse, load_rf
from xgboost_model import predict, load_xgb


def clean_nan(obj):
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


@app.route('/')
def index():
    return render_template(
        'index.html',
        cities=CITIES, genders=GENDER_LABELS,
        weapons=WEAPONS, domains=DOMAINS,
        descriptions=DESCRIPTIONS, age_ranges=AGE_RANGES
    )


@app.route('/api/status')
def status():
    return jsonify({'models_loaded': True})


@app.route('/api/analyse', methods=['POST'])
def api_analyse():
    try:
        data = request.get_json()
        result = analyse(
            city=data.get('city', 'Mumbai'),
            age=int(data.get('age', 25)),
            gender=data.get('gender', 'M'),
            weapon=data.get('weapon', 'None'),
            domain=data.get('domain', 'Other Crime'),
            crime_description=data.get('crime_description', None)
        )
        return jsonify({'success': True, 'data': clean_nan(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        result = predict(
            city=data.get('city', 'Mumbai'),
            age=int(data.get('age', 25)),
            gender=data.get('gender', 'M'),
            weapon=data.get('weapon', 'None'),
            domain=data.get('domain', 'Other Crime'),
            crime_description=data.get('crime_description', None)
        )
        return jsonify({'success': True, 'data': clean_nan(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    load_rf()
    load_xgb()
    print("http://localhost:5000")
    app.run(
        debug=False,
        host='127.0.0.1',   
        port=5000,
        use_reloader=False
    )