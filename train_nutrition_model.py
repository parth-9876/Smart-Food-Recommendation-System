import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# This script trains XGBoost models AND validates the Biomarker Engine,
# printing a side-by-side accuracy comparison for each disease.

print("Loading expanded nutritional data from food_nutrition_data.csv...")
try:
    df = pd.read_csv('food_nutrition_data.csv')
    df.dropna(subset=['Dish Name'], inplace=True)
    print("CSV loaded successfully. Total records:", len(df))
except FileNotFoundError:
    print("\n--- FATAL ERROR ---")
    print("food_nutrition_data.csv not found. Please make sure the file is in the correct folder.")
    print("-------------------\n")
    exit()

# --- 1. Define Features (Nutrients) and Targets (Diseases) ---
features = [
    'Calories', 'Carbohydrate', 'Protein', 'Fats', 'Free Sugar', 'Fibre',
    'Sodium', 'Calcium', 'Iron', 'Vitamin C', 'Folate'
]
diseases = ['diabetes', 'hypertension', 'hyperlipide', 'thyroid']


# ================================================================
# SECTION A: XGBoost Model Training & Evaluation (Original System)
# ================================================================
print("\n" + "=" * 60)
print("XGBOOST MODEL EVALUATION (80/20 Test Split)")
print("=" * 60)

for disease in diseases:
    print(f"\n--- Metrics for: {disease.capitalize()} ---")

    X = df[features]
    y = df[disease]

    # Data Quality Check: Ensure there are both 0s and 1s in the data
    if y.nunique() < 2:
        print(f"Skipping {disease.capitalize()}: The dataset contains only one class (all GOOD or all AVOID). Model cannot be trained.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # THE FIX: Calculate scale_pos_weight to handle class imbalance
    # This tells the model to pay more attention to the rare class (usually 'AVOID').
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1

    # Initialize the XGBoost model WITH the fix and cleaned parameters
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['AVOID (0)', 'GOOD (1)'], zero_division=0))


# --- Train Final XGBoost Models on the Entire Dataset ---
final_models = {}
print("\n--- Training Final XGBoost Models on 100% of the Data ---")

for disease in diseases:
    print(f"Training final model for: {disease.capitalize()}...")

    X_full = df[features]
    y_full = df[disease]

    if y_full.nunique() < 2:
        print(f"Skipping final model for {disease.capitalize()}: Cannot train with only one class.")
        continue

    # Apply the same fix for the final models
    scale_pos_weight_full = np.sum(y_full == 0) / np.sum(y_full == 1) if np.sum(y_full == 1) > 0 else 1

    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_full
    )
    final_model.fit(X_full, y_full)
    final_models[disease] = final_model
    print(f"Final model for {disease.capitalize()} is ready.")

# Save XGBoost Models
joblib.dump(final_models, 'nutrition_models.pkl')
print("\nAll final XGBoost models have been trained and saved to nutrition_models.pkl")


# ================================================================
# SECTION B: Biomarker Engine Evaluation (New System)
# ================================================================
print("\n" + "=" * 60)
print("BIOMARKER ENGINE EVALUATION (Auto-Calibrated on Full Dataset)")
print("=" * 60)

from biomarker_engine import calibrate, compute_biomarkers, evaluate_food_for_disease

# Calibrate on the full dataset (auto-tunes thresholds using labels)
calibrate(df)

# Evaluate accuracy per disease
for disease in diseases:
    if disease not in df.columns:
        continue

    labels = df[disease].values
    if len(set(labels)) < 2:
        print(f"\nSkipping {disease.capitalize()}: only one class in dataset.")
        continue

    predictions = []
    for _, row in df.iterrows():
        bm = compute_biomarkers(row)
        verdict, _ = evaluate_food_for_disease(bm, disease)
        predictions.append(1 if verdict == "GOOD TO EAT" else 0)

    print(f"\n--- Biomarker Metrics for: {disease.capitalize()} ---")
    print(classification_report(
        labels, predictions,
        target_names=['AVOID (0)', 'GOOD (1)'],
        zero_division=0
    ))


# ================================================================
# SECTION C: Side-by-Side Summary
# ================================================================
print("\n" + "=" * 60)
print("SUMMARY: XGBoost vs Biomarker Engine (Full Dataset Accuracy)")
print("=" * 60)

for disease in diseases:
    if disease not in df.columns or df[disease].nunique() < 2:
        continue

    labels = df[disease].values

    # XGBoost full-dataset accuracy (using final trained models)
    if disease in final_models:
        xgb_preds = final_models[disease].predict(df[features])
        xgb_acc = sum(1 for p, l in zip(xgb_preds, labels) if p == l) / len(labels)
    else:
        xgb_acc = None

    # Biomarker accuracy
    bm_preds = []
    for _, row in df.iterrows():
        bm = compute_biomarkers(row)
        verdict, _ = evaluate_food_for_disease(bm, disease)
        bm_preds.append(1 if verdict == "GOOD TO EAT" else 0)
    bm_acc = sum(1 for p, l in zip(bm_preds, labels) if p == l) / len(labels)

    xgb_str = f"{xgb_acc:.1%}" if xgb_acc else "N/A"
    bm_str = f"{bm_acc:.1%}"
    print(f"  {disease.capitalize():20s} | XGBoost: {xgb_str:>6s} | Biomarker: {bm_str:>6s}")

print("\nDone! Both systems evaluated successfully.")
