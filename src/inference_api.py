from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import numpy as np

class TitanicInput(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Sex_male: int
    Embarked_Q: int
    Embarked_S: int

# Initialize FastAPI server
app = FastAPI()

# Load model from MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/titanic_rf_model/1")

@app.post("/predict")
def predict(input: TitanicInput):
    data = np.array([[input.Pclass, input.Age, input.SibSp, input.Parch, input.Fare, input.Sex_male, input.Embarked_Q, input.Embarked_S]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}