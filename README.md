# House Price Prediction using Machine Learning

An end-to-end Machine Learning project that predicts house prices based on property features like area, location attributes, and amenities. This project covers everything from data preprocessing and model training to web deployment using Flask.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat) ![Flask](https://img.shields.io/badge/Flask-2.0-lightgrey) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-RandomForest-orange) ![Deployment](https://img.shields.io/badge/Deployed-Local%20Server-green)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Model Performance](#model-performance)
- [Future Improvements](#future-improvements)
- [Connect with Me](#connect-with-me)

---

## 📖 Overview

This project solves a regression problem where the goal is to estimate the price of a house based on features like:

- Area (sq ft)
- Number of bedrooms & bathrooms
- Road access (mainroad)
- Amenities like guest room, air conditioning, hot water, etc.
- Parking availability and furnishing status

---

## 🎥 Demo

Watch the full walkthrough video on [YouTube](https://youtu.be/WP0hS1CTi5k?si=yYaAOaZVEooZn64P) 

## Screenshots

![Demo Screenshot](https://i.postimg.cc/DzDrRxtq/hpss1.png)
![Demo Screenshot](https://i.postimg.cc/d1y2F8Yg/hpss2.png)

---

## 📁 Project Structure
```bash
├── app.py # Flask backend
├── train_model.py # Model training script
├── housing.csv # Dataset
├── house_price_model.joblib # Trained model
├── scaler.joblib # Saved scaler
├── label_encoders.joblib # Saved label encoders
├── requirements.txt # Required packages
└── templates/
└── index.html # Web UI
```

---

## 🚀 Features

- Categorical variable encoding
- Feature scaling using `StandardScaler`
- Random Forest Regressor for price prediction
- Reusable model, scaler, and encoder artifacts
- Flask web interface and API endpoint
- Real-time predictions

---

## 🧠 Tech Stack

- **Languages:** Python
- **Libraries:** pandas, numpy, scikit-learn, joblib, flask
- **Tools:** HTML/CSS, Flask
- **Model:** Random Forest Regressor

---

## ⚙️ How to Run

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
````

3. **Train the model**

```bash
python train_model.py
````
4. **Run the Flask app**

```bash
python app.py
````
5. **Open in your browser**

```bash
http://127.0.0.1:5000/

````

---

## 📈 Model Performance

```
Metric	Score
Train R²	0.8783
Test R²	0.5917
Model Size	0.17 MB
```

---

## 🔧 Future Improvements

- Deploy the app on Render or Heroku

- Add visual charts for better insights

- Improve UI/UX with form validations

- Use GridSearchCV for better hyperparameter tuning

---

## 🤝 Connect with Me

- 💼 [LinkedIn](https://www.linkedin.com/in/parthiv-majumdar-524046238/)

- 🧠 [YouTube](https://www.youtube.com/@parthivmajumdar6805)

- 💻 [Portfolio](https://portfolio-parthiv.vercel.app/)



