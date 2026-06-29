# PurchaseIQ — E-Commerce Purchase Predictor

A machine learning project that predicts whether an e-commerce customer will make a purchase, using three trained classifiers and an interactive web frontend.

## Project Structure

```
ecom-purchase-predictor/
│
├── data/
│   └── ecom_purchase_prediction4.csv   ← your dataset goes here
│
├── ml/
│   ├── train.py                        ← train all 3 models & save to /models
│   └── predict.py                      ← load models & run predictions
│
├── models/                             ← auto-created after training
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   └── logistic_regression.pkl
│
├── plots/                              ← auto-created after training
│   ├── purchase_distribution.png
│   └── age_distribution.png
│
├── frontend/
│   ├── index.html                      ← web app UI
│   ├── style.css
│   └── app.js
│
├── requirements.txt
└── README.md
```

## Models

| Model               | Algorithm                    | Notes                              |
|---------------------|------------------------------|------------------------------------|
| Random Forest       | Ensemble of Decision Trees   | Most robust; handles noise well    |
| Decision Tree       | Single interpretable tree    | Fast, transparent reasoning        |
| Logistic Regression | Linear + sigmoid probability | Solid baseline; probabilistic      |

## Features Used

| Feature                | Description                        |
|------------------------|------------------------------------|
| `Age`                  | Customer age in years              |
| `Session_Duration_Min` | Time spent on site (minutes)       |
| `Pages_Viewed`         | Number of pages browsed            |
| `Items_In_Cart`        | Number of items added to cart      |
| `Days_Since_Last_Visit`| Recency of last site visit         |
| `Discount_Used`        | Whether a discount code was used   |

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ecom-purchase-predictor.git
cd ecom-purchase-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your dataset

Place `ecom_purchase_prediction4.csv` inside the `data/` folder.

### 4. Train the models

```bash
python ml/train.py
```

This will:
- Print a full accuracy report for each model
- Save `.pkl` model files to `models/`
- Save EDA plots to `plots/`

### 5. Run a prediction from Python

```bash
python ml/predict.py
```

Or import it in your own script:

```python
from ml.predict import load_models, predict

models = load_models()

customer = {
    'Age': 30,
    'Session_Duration_Min': 12,
    'Pages_Viewed': 20,
    'Items_In_Cart': 3,
    'Days_Since_Last_Visit': 10,
    'Discount_Used': 1,
}

results = predict(customer, models)
for model, info in results.items():
    print(f"{model}: {info['label']}")
```

### 6. Open the Web App

Open `frontend/index.html` in your browser — no server required.  
Enter customer values and all three models will vote on the outcome in real time.

> **Note:** The frontend uses a browser-side simulation of the model logic. To connect the real `.pkl` models, add a small Flask or FastAPI backend and replace the `runModels()` function in `app.js` with a `fetch()` call.

## Requirements

```
pandas
scikit-learn
matplotlib
joblib
```

## License

MIT
