# 🚗 Vehicle Price Predictor

![Model](https://img.shields.io/badge/Model-RandomForestRegressor-blue) ![Python](https://img.shields.io/badge/Python-3.12-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

A machine learning system that predicts the **price of a vehicle in USD** based on its specifications like make, model, year, mileage, engine type, transmission, and more.

---

## 🎯 Objective

To accurately estimate a vehicle's price using real-world specifications, enhancing resale platforms, dealership insights, and buyer decisions.

---

## 📁 Dataset Overview

The dataset includes the following features:

- `name`: Vehicle make + model
- `make`, `model`, `year`
- `fuel`, `engine`, `cylinders`, `mileage`, `transmission`, `drivetrain`
- `trim`, `body`, `doors`, `exterior_color`, `interior_color`
- 🎯 `price`: Target variable in USD

---

## ⚙️ Tech Stack

- Python 3.12
- Pandas & NumPy for data handling
- Seaborn & Matplotlib for visualization
- Scikit-learn for modeling (`RandomForestRegressor`)

---

## 🛠️ How It Works

1. Loads & cleans raw vehicle data
2. Encodes categorical fields (e.g., make, model)
3. Scales numeric features
4. Splits into train/test sets
5. Trains a Random Forest Regressor
6. Outputs performance metrics (R², RMSE)
7. Visualizes most important features

---

## 📈 Results

```bash
R2 Score: 0.91
RMSE: $1,427.31
