# Machine Downtime Predictor API

A FastAPI-based machine learning application that predicts machine downtimes in manufacturing processes. The API provides endpoints for training models and making predictions based on manufacturing data.

## 📋 Features

- Data-driven downtime predictions with confidence scores
- Dynamic feature handling based on input dataset
- Interactive API documentation (Swagger UI)
- Model training and persistence
- Real-time predictions

## 🛠️ Tech Stack

- FastAPI (REST API framework)
- scikit-learn (Machine Learning)
- pandas (Data Processing)
- uvicorn (ASGI Server)

## 📦 Installation

1. Clone the repository or download the code files

2. Install required packages:
```bash
pip install fastapi uvicorn pandas numpy scikit-learn python-multipart
```

3. Place your dataset at:
```
C:\Users\Ayaan\OneDrive\Desktop\Fast_API_PROJ\Machine Downtime.csv
```

## 🚀 Running the Application

1. Navigate to the project directory:
```bash
cd C:\Users\Ayaan\OneDrive\Desktop\Fast_API_PROJ
```

2. Run the FastAPI server:
```bash
python -m uvicorn main:app --reload
```

3. The API will be available at:
- Main API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📚 API Endpoints

### 1. Root Endpoint (GET /)
- Access the API landing page with documentation
- URL: http://localhost:8000/

### 2. Features Endpoint (GET /features)
- Get list of available features from the dataset
- URL: http://localhost:8000/features
- Example Response:
```json
{
    "features": ["Machine_ID", "Temperature", "Run_Time"],
    "target": "Downtime_Flag"
}
```

### 3. Train Model Endpoint (POST /train)
- Train the machine learning model
- URL: http://localhost:8000/train
- Example Response:
```json
{
    "accuracy": 0.95,
    "f1_score": 0.94,
    "model_type": "DecisionTreeClassifier"
}
```

### 4. Predict Endpoint (POST /predict)
- Get downtime predictions for new data
- URL: http://localhost:8000/train
- Example Request:
```json
{
    "Machine_ID": 1,
    "Temperature": 85.5,
    "Run_Time": 120.0
}
```
- Example Response:
```json
{
    "Downtime": "Yes",
    "Confidence": 0.89
}
```

## 📁 Project Structure

```
Fast_API_PROJ/
│
├── main.py                 # Main FastAPI application
├── Machine Downtime.csv    # Dataset
└── README.md             # This file
```

## 🔄 Usage Flow

1. Start the server using the instructions above
2. Access http://localhost:8000/docs for interactive API documentation
3. Check available features using the `/features` endpoint
4. Train the model using the `/train` endpoint
5. Make predictions using the `/predict` endpoint with your feature values

## ⚠️ Important Notes

- Ensure your dataset is in CSV format
- The last column of your dataset is assumed to be the target variable
- The model is automatically trained on numeric features
- Model and scaler objects are saved after training
- The server must be restarted if the dataset structure changes

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for any improvements.

## 📝 License

This project is open-source and available under the MIT License.
