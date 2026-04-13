# AI Production Demand Aligner

## Overview
This project is an AI-based system that predicts demand and optimizes production using machine learning and constraints.

## Features
- Demand Forecasting
- Constraint Handling (Labor, Material, Capacity)
- Bottleneck Detection
- Explanation System
- Streamlit Dashboard

## Tech Used
- Python
- Pandas, NumPy
- Streamlit
- Scikit-learn (Random Forest for forecasting)
- PuLP (used for optimization modeling)

## How to Run

1. Install:
pip install pandas numpy scikit-learn streamlit joblib

2. Run:
streamlit run app.py

3. Open:
http://localhost:8501

## Project Files
- app.py → main dashboard
- model_building.ipynb → model building
- final_merged_data.csv → data
- model.pkl → trained model

## Output
- Predicted demand
- Production plan
- Bottleneck
- Explanation
