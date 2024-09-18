from fastapi import FastAPI, HTTPException, Request, status
from app.schema import UserInput
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from fastapi.responses import JSONResponse  # Change response type to JSON

app = FastAPI()

# Load the Ridge model and scaler
model_path = r"C:\Users\ADMIN\PROJECTS\Forest Fires Deployment\models\ridge.pkl"
scaler_path = r"C:\Users\ADMIN\PROJECTS\Forest Fires Deployment\models\scaler.pkl"

with open("models/ridge.pkl", "rb") as model_file:
    ridge_model = pickle.load(model_file)

with open("models/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


@app.post("/predict")
def predict_fire(input_data: UserInput):
    try:
        # Convert the input data to a list for prediction
        data = [
            input_data.temperature,
            input_data.Ws,
            input_data.Rain,
            input_data.FFMC,
            input_data.DMC,
            input_data.ISI,
            input_data.FWI,
            input_data.classes,
            input_data.region,
        ]

        # Scale the data using the loaded scaler
        scaled_data = scaler.transform([data])

        # Perform prediction using the Ridge model
        predicted_rh = ridge_model.predict(scaled_data)[0]

        return JSONResponse({"Predicted RH": predicted_rh})  # Return prediction in JSON

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {e}",
        )

# Removed routes for form and form submission (as there are no templates)