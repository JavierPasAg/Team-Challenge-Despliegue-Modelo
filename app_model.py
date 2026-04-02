from flask import Flask, jsonify, request, render_template
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("models/modelo_optimizado.pkl")

# Landing page
@app.route("/", methods=["GET"])
def hello():
    return render_template("index.html")

# Formulario de predicción
@app.route("/api/v1/predict-form", methods=["GET"])
def predict_form():
    return render_template("form.html")

# Endpoint de predicción
@app.route("/api/v1/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No se ha proporcionado un cuerpo JSON"}), 400

    required_columns = [
        'host_since', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
        'host_listings_count', 'host_total_listings_count', 'host_verifications',
        'host_has_profile_pic', 'host_identity_verified', 'latitude', 'longitude',
        'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'minimum_nights',
        'maximum_nights', 'availability_365', 'number_of_reviews_ltm',
        'estimated_occupancy_l365d', 'review_scores_rating', 'review_scores_accuracy',
        'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value', 'instant_bookable',
        'reviews_per_month', 'host_response_time_ord', 'host_response_time_num',
        'has_host_responded', 'has_reviews', 'days_since_last_review', 'review_lifetime',
        'bathrooms_num', 'is_bathroom_shared', 'room_type_Hotel room', 'room_type_Private room',
        'room_type_Shared room', 'ng_Barajas', 'ng_Carabanchel', 'ng_Centro', 'ng_Chamartín',
        'ng_Chamberí', 'ng_Ciudad Lineal', 'ng_Fuencarral - El Pardo', 'ng_Hortaleza',
        'ng_Latina', 'ng_Moncloa - Aravaca', 'ng_Moratalaz', 'ng_Puente de Vallecas',
        'ng_Retiro', 'ng_Salamanca', 'ng_San Blas - Canillejas', 'ng_Tetuán', 'ng_Usera',
        'ng_Vicálvaro', 'ng_Villa de Vallecas', 'ng_Villaverde', 'neighbourhood_revenue',
        'pt_revenue'
    ]

    input_values = {}
    missing_cols = []

    for col in required_columns:
        if col in data:
            input_values[col] = data[col]
        else:
            input_values[col] = 0.0
            missing_cols.append(col)

    input_data = pd.DataFrame([input_values])[required_columns]

    try:
        prediction = model.predict(input_data)
        result = float(np.expm1(prediction[0]))

        response = {
    "prediction": round(result, 2),
    "status": "success"
}

        if missing_cols:
            response["note"] = f"Se usaron valores por defecto para {len(missing_cols)} campos faltantes."

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
