from flask import Flask, jsonify, request
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
model = joblib.load("modelo_optimizado.pkl")

# os.chdir(os.path.dirname(__file__))

# app = Flask(__name__)

# Carga el modelo
# with open("modelo_optimizado.pkl","rb") as f:
#     model = pickle.load(f)

# Enruta la landing page (endpoint /)
@app.route("/",methods = ['GET'])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    return "Bienvenido a mi API del modelo advertising"

# Enruta la funcion al endpoint /api/v1/predict
@app.route("/api/v1/predict",methods = ['GET'])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET

    
    accommodates = request.args.get('accommodates', np.nan, type=float)
    availability_365 = request.args.get('availability_365', np.nan, type=float)
    bathrooms = request.args.get('bathrooms', np.nan, type=float)

    missing = [name for name, val in [('accommodates', accommodates), ('availability_365', availability_365), ('bathrooms', bathrooms)] if np.isnan(val)]

    input_data = pd.DataFrame({'accommodates': [accommodates], 'availability_365': [availability_365], 'bathrooms': [bathrooms]})
    print("Columnas que espera el modelo:")
    print(model.feature_names_in_)

    print("Columnas que estoy enviando:")
    print(input_data.columns.tolist())
    prediction = model.predict(input_data)

    response = {'predictions': prediction[0]}
    if missing:
        response['warning'] = f"Missing values imputed for: {', '.join(missing)}"

    return jsonify(response)

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
