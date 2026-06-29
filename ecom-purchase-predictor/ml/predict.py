"""
E-Commerce Purchase Prediction — Predictor
==========================================
Load trained models and run predictions on new customer data.
"""

import os
import joblib
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

FEATURES = [
    'Age',
    'Session_Duration_Min',
    'Pages_Viewed',
    'Items_In_Cart',
    'Days_Since_Last_Visit',
    'Discount_Used',
]

MODEL_NAMES = ['random_forest', 'decision_tree', 'logistic_regression']


def load_models() -> dict:
    """Load all saved .pkl models from /models/."""
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f'{name}.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                "Run `python ml/train.py` first to train and save models."
            )
        models[name] = joblib.load(path)
    return models


def predict(customer: dict, models: dict | None = None) -> dict:
    """
    Predict purchase likelihood for a single customer.

    Parameters
    ----------
    customer : dict  — keys must match FEATURES list
    models   : dict  — pre-loaded models (loaded automatically if None)

    Returns
    -------
    dict  — { model_name: { 'prediction': 0|1, 'label': 'WILL/WILL NOT PURCHASE' } }
    """
    if models is None:
        models = load_models()

    df = pd.DataFrame([customer])[FEATURES]

    results = {}
    for name, model in models.items():
        pred = int(model.predict(df)[0])
        results[name] = {
            'prediction': pred,
            'label':      'WILL PURCHASE' if pred == 1 else 'WILL NOT PURCHASE',
        }
    return results


# ── Quick CLI demo ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sample = {
        'Age':                    30,
        'Session_Duration_Min':   12,
        'Pages_Viewed':           20,
        'Items_In_Cart':           3,
        'Days_Since_Last_Visit':  10,
        'Discount_Used':           1,
    }

    print("Customer:", sample)
    print()
    models  = load_models()
    results = predict(sample, models)
    for name, info in results.items():
        label = name.replace('_', ' ').title().ljust(24)
        print(f"  {label}: {info['label']}")
