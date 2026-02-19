<div align="right">
  <a href="README.md">🇹🇷 Türkçe</a> | <a href="README_EN.md">🇬🇧 English</a>
</div>

# 💎 <img src="https://flagcdn.com/w40/gb.png" width="32" alt="EN" style="vertical-align: middle;"> Diamond Price Prediction: End-to-End ML Pipeline & API

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=flat&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat&logo=pandas)

## 📌 Project Overview
This project is a complete, end-to-end machine learning solution designed to predict the price of diamonds based on their physical attributes (carat, cut, color, clarity, depth, table, and dimensions). 

It goes beyond basic model training by implementing a robust data preprocessing pipeline, an optimized **Support Vector Regression (SVR)** model, and deploying the model as a real-time RESTful web service using **FastAPI**.

## ✨ Key Features
* **🧹 Robust Data Preprocessing:** Automated handling of impossible physical dimensions (zero values) and EDA-driven outlier removal.
* **⚙️ Feature Engineering:** Integration of `LabelEncoder` for categorical variables and `StandardScaler` for numerical scaling.
* **🧠 Machine Learning:** Utilizes a highly tuned SVR (Support Vector Regression) with an RBF kernel for accurate price estimation.
* **🚀 Real-Time API:** A fully functional web interface and REST API built with FastAPI to serve predictions instantly.
* **📦 Model Serialization:** End-to-end pipeline (Model + Scaler + Encoders) saved securely via `pickle` for seamless deployment.

## 🛠️ Tech Stack
* **Data Science:** `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (SVR, Preprocessing, Metrics)
* **Web Deployment:** `FastAPI`, `Uvicorn`, `Jinja2`

## 📂 Project Structure
```text
├── train_model.py         # Data loading, cleaning, and model training script
├── main.py                # FastAPI web server and prediction endpoint
├── 10-diamonds.csv        # Raw dataset
├── diamond_model_complete.pkl # Serialized model and preprocessors (Generated)
├── testdata.csv           # Test dataset for evaluation (Generated)
├── requirements.txt       # Project dependencies
└── templates/
    └── index.html         # Frontend interface for the API
```
## 💻 Installation & Usage



1.  Clone the repository:

    ```bash

    git clone https://github.com/AdnanSag/ml-deployment-fastapi-diamond.git

    ```

2.  Install dependencies:

    ```bash

    pip install -r requirements.txt

    ```

---



## 📬 Contact



If you'd like to talk about my projects or collaborate, feel free to reach out:

* **LinkedIn:** https://www.linkedin.com/in/adnan-sag/ 

* **Email:** adnansag91@gmail.com



*Created by Adnan Sag* 
