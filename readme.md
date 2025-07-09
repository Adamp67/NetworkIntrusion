# Network Intrusion Detection System (NIDS)

This Streamlit app is a machine learning-powered Network Intrusion Detection System. It allows users to upload traffic data and simulate real-time or batch detection of cyber attacks using trained models (Random Forest, Logistic Regression, KNN).

## Features

-  Upload network data or use demo
-  Choose between 3 ML models
-  Live traffic simulation for Random Forest
-  Model comparison (accuracy, F1, AUC, etc.)
-  Educates users on cybersecurity principles

## How to Use

1. Upload a `.csv` dataset (or use demo)
2. Choose a detection model
3. Run detection or enable simulation
4. View model insights

## Note
Model files (`.pkl`) and datasets are excluded from GitHub via `.gitignore`. Place them manually inside `streamlit_app/Models` for local use.
