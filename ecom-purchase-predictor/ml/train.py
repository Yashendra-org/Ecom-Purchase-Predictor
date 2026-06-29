"""
E-Commerce Purchase Prediction — Model Training
================================================
Trains 3 classifiers (Random Forest, Decision Tree, Logistic Regression)
on customer behavioural data and saves the models to /models/.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'ecom_purchase_prediction4.csv')
MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'plots')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

FEATURES = [
    'Age',
    'Session_Duration_Min',
    'Pages_Viewed',
    'Items_In_Cart',
    'Days_Since_Last_Visit',
    'Discount_Used',
]
TARGET = 'Purchased'


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅ Loaded {len(df):,} rows from {path}")
    print(df[FEATURES + [TARGET]].describe())
    return df


# ── EDA Plots ────────────────────────────────────────────────────────────────
def save_plots(df: pd.DataFrame) -> None:
    # Purchase distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    df[TARGET].value_counts().plot(kind='bar', color='steelblue', ax=ax)
    ax.set_title('Did Customer Buy?')
    ax.set_xlabel('0 = No  |  1 = Yes')
    ax.set_ylabel('Number of Customers')
    ax.set_xticklabels(['No', 'Yes'], rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'purchase_distribution.png'), dpi=150)
    plt.close(fig)

    # Age distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df['Age'], color='orange', edgecolor='white', bins=20)
    ax.set_title('Customer Age Distribution')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'age_distribution.png'), dpi=150)
    plt.close(fig)

    print("📊 Plots saved to /plots/")


# ── Training ─────────────────────────────────────────────────────────────────
MODELS = {
    'random_forest':     RandomForestClassifier(random_state=42),
    'decision_tree':     DecisionTreeClassifier(random_state=42),
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
}

def train_all(df: pd.DataFrame) -> dict:
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n📂 Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    results = {}
    for name, model in MODELS.items():
        print(f"--- Training: {name.replace('_', ' ').title()} ---")
        model.fit(X_train, y_train)
        preds    = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        results[name] = {'model': model, 'accuracy': accuracy}
        print(f"Accuracy : {accuracy * 100:.2f}%")
        print(classification_report(y_test, preds, target_names=['No Buy', 'Buy']))

    return results


# ── Save Models ───────────────────────────────────────────────────────────────
def save_models(results: dict) -> None:
    for name, info in results.items():
        path = os.path.join(MODELS_DIR, f'{name}.pkl')
        joblib.dump(info['model'], path)
        print(f"💾 Saved: {path}")


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(results: dict) -> None:
    best = max(results, key=lambda k: results[k]['accuracy'])
    print("\n" + "=" * 50)
    print("           FINAL ACCURACY SUMMARY")
    print("=" * 50)
    for name, info in results.items():
        tag = " ← best" if name == best else ""
        label = name.replace('_', ' ').title().ljust(24)
        print(f"  {label}: {info['accuracy'] * 100:.2f}%{tag}")
    print("=" * 50 + "\n")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df      = load_data(DATA_PATH)
    save_plots(df)
    results = train_all(df)
    save_models(results)
    print_summary(results)
