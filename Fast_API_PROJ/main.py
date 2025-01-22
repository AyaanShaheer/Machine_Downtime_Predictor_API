# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, create_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
from typing import List, Optional, Type
import io

# Define the path to your CSV file
DATA_PATH = r"C:\Users\Ayaan\OneDrive\Desktop\Fast_API_PROJ\Machine Downtime.csv"

app = FastAPI(
    title="Manufacturing Downtime Predictor",
    description="An API for predicting machine downtimes in manufacturing processes",
    version="1.0.0"
)

# Global variables to store model and preprocessing objects
model = None
scaler = None
feature_columns = None
target_column = None

# Base class for dynamic model creation
class PredictionInputBase(BaseModel):
    pass

# Initialize PredictionInput as the base class
PredictionInput: Type[PredictionInputBase] = PredictionInputBase

class PredictionOutput(BaseModel):
    Downtime: str
    Confidence: float

class TrainingMetrics(BaseModel):
    accuracy: float
    f1_score: float
    model_type: str

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint providing API information and usage instructions."""
    return """
    <html>
        <head>
            <title>Machine Downtime Predictor API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; }
                .endpoint { background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
                code { background-color: #e9ecef; padding: 2px 4px; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h1>🏭 Machine Downtime Predictor API</h1>
            
            <h2>About</h2>
            <p>This API provides machine learning-based predictions for manufacturing machine downtimes.</p>
            
            <h2>Available Endpoints</h2>
            
            <div class="endpoint">
                <h3>📊 GET /features</h3>
                <p>Returns the list of features used by the model</p>
            </div>
            
            <div class="endpoint">
                <h3>🔧 POST /train</h3>
                <p>Trains the machine learning model and returns performance metrics</p>
            </div>
            
            <div class="endpoint">
                <h3>🎯 POST /predict</h3>
                <p>Makes downtime predictions based on input features</p>
            </div>
            
            <h2>How to Use</h2>
            <ol>
                <li>First, check available features using <code>GET /features</code></li>
                <li>Train the model using <code>POST /train</code></li>
                <li>Make predictions using <code>POST /predict</code> with your feature values</li>
            </ol>
            
            <h2>Documentation</h2>
            <p>For detailed API documentation and interactive testing, visit: 
               <a href="/docs">Swagger UI Documentation</a></p>
        </body>
    </html>
    """




def load_and_validate_data():
    """Load and validate the CSV data."""
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail=f"Data file not found at {DATA_PATH}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error reading data: {str(e)}"
        )

# Initialize the API by reading the data and setting up the models
@app.on_event("startup")
async def startup_event():
    global feature_columns, target_column, PredictionInput
    
    # Load the initial dataset
    df = load_and_validate_data()
    
    # Assuming the last column is the target variable
    feature_columns = df.columns[:-1].tolist()
    target_column = df.columns[-1]
    
    # Create field definitions for dynamic model
    field_definitions = {}
    for col in feature_columns:
        if df[col].dtype in ['float64', 'float32']:
            field_definitions[col] = (float, ...)
        else:
            field_definitions[col] = (int, ...)
    
    # Create dynamic PredictionInput model
    PredictionInput = create_model('PredictionInput', **field_definitions, __base__=PredictionInputBase)

@app.post("/train", response_model=TrainingMetrics)
async def train_model():
    """Train the machine learning model."""
    try:
        # Load data
        df = load_and_validate_data()
        
        # Prepare features and target
        X = df[feature_columns]
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        global scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        global model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test_scaled)
        metrics = TrainingMetrics(
            accuracy=float(accuracy_score(y_test, y_pred)),
            f1_score=float(f1_score(y_test, y_pred)),
            model_type="DecisionTreeClassifier"
        )
        
        # Save model and scaler
        joblib.dump(model, 'model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInputBase):
    """Make predictions using the trained model."""
    try:
        if model is None:
            raise HTTPException(
                status_code=400,
                detail="Model not trained. Please train the model first."
            )
        
        # Prepare input data
        input_df = pd.DataFrame([input_data.dict()])
        
        # Ensure all required features are present
        if not all(col in input_df.columns for col in feature_columns):
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features. Required features: {feature_columns}"
            )
        
        # Scale the input data
        input_scaled = scaler.transform(input_df[feature_columns])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        confidence = max(model.predict_proba(input_scaled)[0])
        
        return PredictionOutput(
            Downtime="Yes" if prediction == 1 else "No",
            Confidence=float(confidence)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features")
async def get_features():
    """Get the list of required features for prediction."""
    return {
        "features": feature_columns,
        "target": target_column
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)