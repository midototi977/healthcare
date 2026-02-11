from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

# =====================
# Load Trained Model
# =====================

with open("medical_healthcare_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "🚀 Healthcare Medical Condition API Running!"}


class InputData(BaseModel):
    Age: float
    Gender: str
    Glucose: float
    Blood_Pressure: float
    BMI: float
    Oxygen_Saturation: float
    LengthOfStay: float
    Cholesterol: float
    Triglycerides: float
    HbA1c: float
    Smoking: float
    Alcohol: float
    Physical_Activity: float
    Diet_Score: float
    Family_History: float
    Stress_Level: float
    Sleep_Hours: float


@app.post("/predict")
def predict(data: InputData):
    try:
        input_dict = data.dict()

        # رجّع أسماء الأعمدة الأصلية بمسافات
        rename_map = {
            "Blood_Pressure": "Blood Pressure",
            "Oxygen_Saturation": "Oxygen Saturation",
            "Physical_Activity": "Physical Activity",
            "Diet_Score": "Diet Score",
            "Family_History": "Family History",
            "Stress_Level": "Stress Level",
            "Sleep_Hours": "Sleep Hours"
        }

        for k, v in rename_map.items():
            input_dict[v] = input_dict.pop(k)

        df = pd.DataFrame([input_dict])

        prediction = model.predict(df)
        probability = model.predict_proba(df)

        return {
            "prediction": prediction[0],
            "probability": float(probability[0].max())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
