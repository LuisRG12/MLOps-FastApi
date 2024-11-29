from pandas import DataFrame
from joblib import load
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException

model = load("pipeline.joblib")

app = FastAPI()

class DataPredict(BaseModel):
    data_to_predict: list[list]  # Lista de listas con las características

columns = ["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", 
           "thalachh", "exng", "oldpeak", "slp", "caa", "thall"]

@app.post("/predict")
def predict(request: DataPredict):
    try:
        list_data = request.data_to_predict
        df_data = DataFrame(list_data, columns=columns)

        prediction = model.predict(df_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {'message': 'Heart Attack Prediction API is running'}
