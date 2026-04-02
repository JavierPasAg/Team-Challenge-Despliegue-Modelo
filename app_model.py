from flask import Flask, jsonify, request, render_template
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

from flask import Flask
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

app = Flask(__name__)

# Carga el modelo
model = joblib.load("models/modelo_optimizado.pkl")

# os.chdir(os.path.dirname(__file__))

# app = Flask(__name__)

# Carga el modelo
# with open("modelo_optimizado.pkl","rb") as f:
#     model = pickle.load(f)

# Enruta la landing page (endpoint /)
@app.route("/",methods = ['GET'])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    return render_template("index.html")

# Enrutamos la función para rellenar los valores con los que predecir
@app.route("/api/v1/predict-form", methods=['GET'])
def predict_form():
    return render_template("form.html")

# Enruta la funcion al endpoint /api/v1/predict
@app.route("/api/v1/predict", methods=['POST'])
def predict():
    # 1. Obtener los datos del cuerpo JSON
    data = request.get_json()
    if not data:
        return jsonify({"error": "No se ha proporcionado un cuerpo JSON"}), 400

    # 2. Tu lista de columnas obligatorias (la he guardado en una variable)
    required_columns = [
        'host_since', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 
        'host_listings_count', 'host_total_listings_count', 'host_verifications', 
        'host_has_profile_pic', 'host_identity_verified', 'latitude', 'longitude', 
        'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'minimum_nights', 
        'maximum_nights', 'availability_365', 'number_of_reviews_ltm', 
        'estimated_occupancy_l365d', 'review_scores_rating', 'review_scores_accuracy', 
        'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 
        'review_scores_location', 'review_scores_value', 'instant_bookable', 
        'reviews_per_month', 'host_response_time_ord', 'has_reviews', 
        'days_since_last_review', 'review_lifetime', 'host_response_time_num', 
        'has_host_responded', 'room_type_Hotel room', 'room_type_Private room', 
        'room_type_Shared room', 'bathrooms_num', 'is_bathroom_shared', 'ng_Barajas', 
        'ng_Carabanchel', 'ng_Centro', 'ng_Chamartín', 'ng_Chamberí', 'ng_Ciudad Lineal', 
        'ng_Fuencarral - El Pardo', 'ng_Hortaleza', 'ng_Latina', 'ng_Moncloa - Aravaca', 
        'ng_Moratalaz', 'ng_Puente de Vallecas', 'ng_Retiro', 'ng_Salamanca', 
        'ng_San Blas - Canillejas', 'ng_Tetuán', 'ng_Usera', 'ng_Vicálvaro', 
        'ng_Villa de Vallecas', 'ng_Villaverde', 'neighbourhood_revenue', 'pt_revenue'
    ]

    # 3. Construir el diccionario de entrada con valores por defecto (0.0)
    # Si la columna está en el JSON, usamos ese valor; si no, ponemos 0.0
    input_values = {}
    missing_cols = []
    
    for col in required_columns:
        if col in data:
            input_values[col] = data[col]
        else:
            input_values[col] = 0.0  # Valor por defecto
            missing_cols.append(col)

    # 4. Crear DataFrame asegurando el ORDEN exacto de las columnas
    input_data = pd.DataFrame([input_values])[required_columns]

    # 5. Realizar la predicción
    try:
        prediction = model.predict(input_data)
        
        # 6. Convertir el resultado a float nativo de Python para evitar el error de JSON
        result = float(prediction[0])
        
        response = {
            'prediction': result,
            'status': 'success'
        }
        
        # Opcional: Avisar si faltaron campos y se usaron ceros
        if missing_cols:
            response['note'] = f"Se usaron valores por defecto para {len(missing_cols)} campos faltantes."

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Enruta la funcion al endpoint /api/v1/retrain
@app.route("/api/v1/retrain/",methods = ['GET'])
def retrain(): # Ligado al endpoint '/api/v1/retrain/', método GET
    global model
    if os.path.exists("data/Advertising_new.csv"):
        data = pd.read_csv('data/Advertising_new.csv')
        data.columns = [col.lower() for col in data.columns]

        X = data.drop(columns=['sales'])
        y = data['sales']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(X, y)

        with open('ad_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

if __name__ == '__main__':
    app.run(debug=True)
