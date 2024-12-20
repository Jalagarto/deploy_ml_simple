from fastapi import FastAPI
import joblib
from pydantic import BaseModel

MODEL_PATH = "/app/model_pipeline.joblib"
model = joblib.load(MODEL_PATH)

# Define your FastAPI app
app = FastAPI()


# Input schema for predictions
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float


# Map integer labels to string labels
LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}


@app.post("/predict/")
async def predict(data: InputData):
    # Extract input data
    input_data = [[data.feature1, data.feature2, data.feature3, data.feature4]]
    # Make a prediction
    prediction = model.predict(input_data)
    # Map the integer prediction to a string label
    string_prediction = LABELS.get(prediction[0], "Unknown")
    return {"prediction": string_prediction}
