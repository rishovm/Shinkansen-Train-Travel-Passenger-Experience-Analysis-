# Shinkansen-Train-Travel-Passenger-Experience-Analysis-
The goal of the problem is to predict whether a passenger was satisfied or not considering his/her overall experience of traveling on the Shinkansen Bullet Train.
🚄 Shinkansen Travel Experience – Predictive Analytics Project

Author: [Your Name]
Location: Amsterdam, Netherlands
Tools: Python · XGBoost · LightGBM · Random Forest · TensorFlow · Scikit-learn · Matplotlib · Pandas

🧠 Project Overview

This project analyzes and predicts passenger travel satisfaction on Japan’s Shinkansen (bullet train) network.
The goal was to build a robust end-to-end machine learning pipeline to identify factors influencing customer experience and predict satisfaction outcomes with high accuracy.

Developed during a hackathon, the project combines:

Data cleaning and preprocessing

Feature engineering and model training (XGBoost, LightGBM, Random Forest, Neural Network)

Evaluation and model comparison

Automated visualization of model performance

Scalable saving/loading of trained models

🎯 Objectives

Understand what drives passenger satisfaction (e.g. seat comfort, Wi-Fi, punctuality).

Predict whether a traveler is satisfied or not.

Compare multiple machine learning models for best performance.

Automate training, validation, and model saving pipelines for reproducibility.

⚙️ Key Features

✅ Data Pipeline

Cleans missing values and handles categorical encoding

Scales/normalizes continuous variables

Splits data into train/validation/test sets

✅ Model Suite

Model	Framework	Purpose
XGBoost	xgboost	Baseline gradient boosting model
LightGBM	lightgbm	Fast, efficient boosting with early stopping
Random Forest	sklearn	Baseline ensemble for interpretability
Neural Network	TensorFlow / Keras	Deep learning model for non-linear relationships

✅ Training Configuration

Early stopping to prevent overfitting

Automated hyperparameter tuning (learning rate, regularization, leaves, etc.)

Model performance tracked via logloss and accuracy metrics

✅ Automation

All models saved to /Saved_Models using joblib and Keras’ native .keras format

Flexible code structure for retraining with new data

✅ Visualization

Training/validation loss curves

Comparative evaluation metrics

Optional notebook workflow graph (via Graphviz)

🧩 Tech Stack
Category	Tools Used
Language	Python 3.12
Core ML	XGBoost · LightGBM · RandomForest · TensorFlow/Keras
Data Handling	Pandas · NumPy
Evaluation	Scikit-learn metrics
Visualization	Matplotlib · Graphviz
Persistence	Joblib · Keras save models
Environment	Google Colab / Jupyter Notebook
📊 Model Training Summary
Model	Validation LogLoss	Notes
XGBoost	~0.099	Best performing model
LightGBM	~0.100	Stable generalization
Random Forest	Moderate	Used as baseline
Neural Network	Stable	SHAP-free implementation
📁 Repository Structure
📦 Shinkansen-Travel-Experience
│
├── 📜 Shinkansen_Travel_Experience_Rishov.ipynb      # Main analysis notebook
├── 📂 Saved_Models/                                  # Serialized models
│   ├── xgb_model.pkl
│   ├── lgb_model.pkl
│   ├── rf_model.pkl
│   └── nn_model.keras
├── 📜 requirements.txt                                # Environment dependencies
├── 📜 README.md                                       # You are here
└── 📜 visualize_notebook_flow.py                      # Optional visualization tool

🧪 How to Run

Clone the repository:

git clone https://github.com/<rishovm>/Shinkansen-Travel-Experience.git
cd Shinkansen-Travel-Experience


Install dependencies:

pip install -r requirements.txt


Open the notebook in Google Colab or Jupyter:

jupyter notebook Shinkansen_Travel_Experience_Rishov.ipynb


Run all cells to:

Preprocess data

Train models


📜 License

This project is released under the MIT License.
You’re free to use, modify, and distribute it with proper credit.
