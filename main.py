from pandas import DataFrame
from joblib import load
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException

model = load("pipeline.joblib")

app = FastAPI()

class DataPredict(BaseModel):
    data_to_predict: list[list] = [
        [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1],  # Ejemplo 1
        [45, 0, 2, 130, 250, 0, 1, 140, 0, 1.5, 1, 2, 2],  # Ejemplo 2
        [54, 1, 1, 120, 220, 0, 0, 160, 1, 1.8, 2, 1, 3]   # Ejemplo 3
    ]

columns = ["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", 
           "thalachh", "exng", "oldpeak", "slp", "caa", "thall"]

@app.post("/predict")
def predict(request: DataPredict):
    try:
        
        list_data = request.data_to_predict
        df_data = DataFrame(list_data, columns=columns)
        prediction = model.predict(df_data)
        interpretations = [
            "Paciente con riesgo de ataque cardiaco" if pred == 1 else "Paciente sin riesgo de ataque cardiaco"
            for pred in prediction
        ]
        return {
            "prediction": prediction.tolist(),
            "interpretations": interpretations
        }
    except Exception as e:
        # Manejo de errores
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {'message': 'Heart Attack Prediction API is running'}
