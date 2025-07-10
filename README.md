# 🚗 Car Price Prediction using Linear Regression

This project uses a **machine learning pipeline** to predict the selling price of used cars based on various features such as age, mileage, fuel type, and transmission. The model is trained using **linear regression** and the **scikit-learn pipeline API**.

---

## 📁 Dataset

The dataset is a CSV file (`car data.csv`) with columns like:

- `Car_Name`
- `Year`
- `Selling_Price` *(target variable)*
- `Present_Price`
- `Driven_kms`
- `Fuel_Type`
- `Selling_type`
- `Transmission`
- `Owner`

---

## 🧠 Project Overview

The notebook/script performs the following steps:

- Loads and previews the dataset
- Creates a new `Car_Age` feature
- Drops unused or redundant columns (`Year`)
- Applies one-hot encoding to categorical variables
- Standardizes numerical features
- Builds a preprocessing and regression pipeline
- Splits the data into training and test sets
- Trains a **Linear Regression** model
- Evaluates the model using RMSE and R²
- Visualizes predicted vs actual prices

---

## 📊 Model Evaluation

- **Metric Used**:
  - RMSE (Root Mean Squared Error)
  - R² Score (Coefficient of Determination)

Sample output:
```
Model Performance on Test Set:
RMSE: 1.65
R^2: 0.91
```

---

## 📈 Visualization

A scatter plot is generated showing **Actual vs Predicted Selling Prices**, with a reference diagonal line to compare predictions.

---

## 🚀 How to Run

1. Clone the repo or download the script.
2. Make sure the file `car data.csv` is in the same folder as your script.
3. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run the script:

```bash
python your_script_name.py
```

---

## ✅ Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## 📌 Notes

- Make sure to replace `your_script_name.py` with your actual filename (e.g., `car_price_predictor.py`)
- This project is based on a supervised regression approach, not suitable for classification tasks.

---

## 📃 License

This project is free to use for educational and personal learning purposes.
