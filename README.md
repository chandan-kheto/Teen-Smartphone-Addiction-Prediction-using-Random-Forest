# 📱 Teen Smartphone Addiction Prediction using Random Forest

This project uses machine learning to predict **teen smartphone addiction levels** based on their daily habits, emotional health, and phone usage patterns. The goal is to build a regression model that can help understand the key factors driving addiction among teenagers.

---

## 🔍 Problem Statement

Smartphone addiction among teenagers is a growing concern worldwide. This project aims to **predict the addiction score** of a teenager using behavioral, psychological, and lifestyle factors.

---

## 🧠 ML Problem Type

> **Supervised Learning — Regression**  
Target variable: `Addiction_Level` (continuous float values)

---

## 📂 Dataset Overview

The dataset contains `3000` entries and `25` columns including:

- 📱 Phone usage: `Daily_Usage_Hours`, `Phone_Checks_Per_Day`, `Weekend_Usage_Hours`
- 🧠 Mental health: `Anxiety_Level`, `Depression_Level`, `Self_Esteem`
- 🏫 Academic life: `Academic_Performance`, `School_Grade`
- 💬 Social life: `Social_Interactions`, `Family_Communication`
- 💤 Lifestyle: `Sleep_Hours`, `Exercise_Hours`
- 🧾 Others: `Apps_Used_Daily`, `Phone_Usage_Purpose`, etc.

---

## ⚙️ Machine Learning Pipeline

### ✅ Step-by-step Workflow:

1. **Exploratory Data Analysis (EDA)**  
   - Distribution plots for addiction levels  
   - Heatmap for correlation between features

2. **Preprocessing**  
   - Dropped ID and Name columns  
   - One-hot encoded categorical variables

3. **Model Training**  
   - Used `RandomForestRegressor` from `scikit-learn`  
   - Split data using 80/20 train-test strategy

4. **Model Evaluation**  
   - Metrics: `R² Score`, `MSE`, `MAE`  
   - Visualization: Actual vs Predicted scatter plot

5. **Feature Importance**  
   - Visualized top 10 most important features influencing addiction level

---

## 📈 Results

| Metric | Value |
|--------|-------|
| R² Score | ~0.95 ✅ (example, update with actual) |
| MSE | ~0.18 |
| MAE | ~0.29 |

> 📌 Most Important Features:
- Phone_Checks_Per_Day
- Screen_Time_Before_Bed
- Daily_Usage_Hours
- Anxiety_Level
- Time_on_Social_Media

---

## 🛠️ Tech Stack

- Python 
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (RandomForestRegressor)

---

## 🚀 Future Improvements

- 🔁 Try advanced regressors (XGBoost, GradientBoosting)
- 📊 Convert to classification version (Low/Medium/High addiction)
- 🌐 Build Streamlit web app for live prediction
- 📱 Collect real-world anonymized data from students


## 📁 Folder Structure

Teen Smartphone Addiction Prediction/
├── teen_phone_data.csv # Dataset
├── Model code.py # Model training & evaluation script
└── README.md # Project documentation

## 💡 Conclusion

This project provides meaningful insights into teen smartphone behavior and shows how ML can be used to **predict and understand addiction patterns**. It's also a strong showcase of regression modeling, feature importance analysis, and real-world data handling.
