# Vehicle Rental Price Prediction

This project predicts the daily rental price of a vehicle using machine learning.

## Project Components

- Data analysis and machine learning notebook
- Cleaned vehicle rental dataset
- Tuned Gradient Boosting model
- Streamlit prediction dashboard
- Data Analysis report
- Machine Learning report

## Final Model Performance

- Model: Tuned Gradient Boosting
- MAE: 32.43
- RMSE: 72.85
- R² Score: 0.3838

## Project Structure

```text
Vehicle_Rental_Price_Prediction/
├── Dashboard/
│   └── app.py
├── DataSet/
├── Models/
│   ├── vehicle_rental_price_pipeline.joblib
│   ├── model_metadata.json
│   └── dashboard_input_options.json
├── Reports/
├── requirements.txt
└── README.md

Setup Instructions
1. Create a virtual environment
    py -m venv venv

2. Activate the virtual environment
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1


3. Install the required libraries
       python -m pip install -r requirements.txt


4. Run the dashboard
    python -m streamlit run Dashboard/app.py
The dashboard will normally open at:

http://localhost:8501

Dashboard Features
Accepts vehicle and location information
Uses the trained machine learning pipeline
Predicts the estimated daily rental price
Displays model evaluation metrics
Shows the entered input values
Dataset

The project uses the CarRentalData dataset with 5,851 vehicle rental records.

Main Libraries
Pandas
NumPy
Scikit-learn
Streamlit
Joblib
# vehicle-rental-price-prediction
