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
