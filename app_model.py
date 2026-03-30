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
# Enruta la funcion al endpoint /api/v1/predict
@app.route("/api/v1/predict", methods=['GET'])
def predict():
    # 1. Obtenemos la lista de todas las columnas que el modelo espera
    # (Esto evita escribir una por una si son muchas)
    expected_columns = model.feature_names_in_

    # 2. Definimos los valores por defecto (ejemplo: 0 o np.nan)
    # Puedes personalizar esto si algunas columnas requieren una media específica
    default_value = 0.0 
    
    # 3. Creamos un diccionario con todos los campos inicializados
    input_values = {col: default_value for col in expected_columns}

    # 4. Actualizamos el diccionario con los parámetros que vengan en la URL
    # Si el usuario no envía 'bedrooms', se queda el 0.0 del paso anterior
    for col in expected_columns:
        val = request.args.get(col)
        if val is not None:
            try:
                input_values[col] = float(val)
            except ValueError:
                pass # O manejar error si el dato no es numérico

    # 5. Convertimos a DataFrame asegurando el orden correcto de las columnas
    input_data = pd.DataFrame([input_values])[expected_columns]

    # 6. Predicción
    

    prediction = model.predict(input_data)

    return jsonify({
        'predictions': prediction.tolist()[0] # Convierte el array a una lista de Python
    })

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
