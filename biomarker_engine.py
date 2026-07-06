"""
Biomarker Engine — The Abstraction Layer Between Nutrition and Disease.

This module computes universal biomarker scores from raw nutritional data,
then evaluates whether a food is safe for a given disease based on configurable
disease profiles. Adding a new disease requires only a new entry in DISEASE_PROFILES.

Architecture:
    Nutrition (11 features) → Biomarker Scoring → Disease Risk Profile → Verdict
"""

import numpy as np

# --- Feature Configuration ---
FEATURE_NAMES = [
    'Calories', 'Carbohydrate', 'Protein', 'Fats', 'Free Sugar',
    'Fibre', 'Sodium', 'Calcium', 'Iron', 'Vitamin C', 'Folate'
]

NUTRIENT_UNITS = {
    'Calories': 'kcal', 'Carbohydrate': 'g', 'Protein': 'g', 'Fats': 'g',
    'Free Sugar': 'g', 'Fibre': 'g', 'Sodium': 'mg', 'Calcium': 'mg',
    'Iron': 'mg', 'Vitamin C': 'mg', 'Folate': 'mcg',
}

# --- Biomarker Metadata ---
BIOMARKER_INFO = {
    'glycemic_impact': {
        'display': 'Glycemic Impact',
        'description': 'Blood sugar spike risk from sugars and carbohydrates',
        'key_nutrients': ['Free Sugar', 'Carbohydrate', 'Fibre'],
    },
    'cardiovascular_strain': {
        'display': 'Cardiovascular Strain',
        'description': 'Heart and blood pressure load from sodium and fats',
        'key_nutrients': ['Sodium', 'Fats'],
    },
    'lipid_load': {
        'display': 'Lipid Load',
        'description': 'Cholesterol and fat accumulation risk',
        'key_nutrients': ['Fats', 'Calories'],
    },
    'inflammatory_index': {
        'display': 'Inflammatory Index',
        'description': 'Chronic inflammation risk from poor nutrient balance',
        'key_nutrients': ['Calories', 'Fibre', 'Vitamin C'],
    },
    'thyroid_stress': {
        'display': 'Thyroid Stress',
        'description': 'Thyroid function strain from nutrient deficiencies',
        'key_nutrients': ['Protein', 'Iron', 'Calcium'],
    },
    'caloric_density': {
        'display': 'Caloric Density',
        'description': 'Energy density and weight gain potential',
        'key_nutrients': ['Calories', 'Fats'],
    },
}

# --- Disease Profiles ---
# Each disease maps to a set of weighted biomarkers and a risk threshold.
# The threshold is auto-calibrated at startup if labeled data exists.
# To add a NEW disease, simply add a new entry here — no retraining needed.
DISEASE_PROFILES = {
    'diabetes': {
        'display': 'Diabetes',
        'weights': {'glycemic_impact': 0.70, 'caloric_density': 0.30},
        'threshold': 0.35,
    },
    'hypertension': {
        'display': 'Hypertension',
        'weights': {'cardiovascular_strain': 0.75, 'inflammatory_index': 0.25},
        'threshold': 0.20,
    },
    'hyperlipide': {
        'display': 'Hyperlipidemia',
        'weights': {'lipid_load': 0.75, 'cardiovascular_strain': 0.25},
        'threshold': 0.35,
    },
    'thyroid': {
        'display': 'Thyroid Disorder',
        'weights': {'thyroid_stress': 0.65, 'inflammatory_index': 0.35},
        'threshold': 0.90,  # Very high — all foods in dataset are GOOD for thyroid
    },
    'obesity': {
        'display': 'Obesity',
        'weights': {'caloric_density': 0.50, 'lipid_load': 0.30, 'glycemic_impact': 0.20},
        'threshold': 0.40,
    },
    'kidney_disease': {
        'display': 'Kidney Disease',
        'weights': {'cardiovascular_strain': 0.80, 'inflammatory_index': 0.20},
        'threshold': 0.25,
    },
    'pcod': {
        'display': 'PCOD/PCOS',
        'weights': {'glycemic_impact': 0.45, 'inflammatory_index': 0.30, 'caloric_density': 0.25},
        'threshold': 0.35,
    },
}

# --- Module State (set during calibration) ---
_feature_min = {}
_feature_max = {}
_calibrated = False


def calibrate(df):
    """
    Initialize the biomarker engine with dataset statistics.
    
    This function:
    1. Computes feature min/max for normalization
    2. Auto-tunes disease thresholds using labeled data (if available)
    
    Args:
        df: pandas DataFrame with FEATURE_NAMES columns and optionally disease label columns.
    """
    global _feature_min, _feature_max, _calibrated, DISEASE_PROFILES

    # Step 1: Compute feature min/max for normalization
    for feat in FEATURE_NAMES:
        _feature_min[feat] = float(df[feat].min())
        _feature_max[feat] = float(df[feat].max())

    _calibrated = True

    # Step 2: Compute biomarkers for all foods in dataset
    all_biomarkers = []
    for _, row in df.iterrows():
        bm = compute_biomarkers(row)
        all_biomarkers.append(bm)

    # Step 3: Auto-tune thresholds for diseases with labels
    for disease, profile in DISEASE_PROFILES.items():
        if disease not in df.columns:
            continue

        labels = df[disease].values
        if len(set(labels)) < 2:
            continue  # Can't calibrate with only one class

        # Compute weighted risk scores for all foods
        weights = profile['weights']
        risk_scores = [
            sum(bm[b] * w for b, w in weights.items())
            for bm in all_biomarkers
        ]

        # Grid search for optimal threshold
        best_threshold = profile['threshold']
        best_accuracy = 0.0

        for thresh_int in range(50, 950, 5):
            thresh = thresh_int / 1000.0
            predictions = [0 if s > thresh else 1 for s in risk_scores]
            accuracy = sum(1 for p, l in zip(predictions, labels) if p == l) / len(labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = thresh

        DISEASE_PROFILES[disease]['threshold'] = best_threshold
        DISEASE_PROFILES[disease]['_calibrated_accuracy'] = best_accuracy

    print("Biomarker engine calibrated successfully.")
    for disease, profile in DISEASE_PROFILES.items():
        acc = profile.get('_calibrated_accuracy')
        acc_str = f"{acc:.1%}" if acc is not None else "N/A (no labels)"
        print(f"  {profile['display']:20s} | threshold: {profile['threshold']:.3f} | accuracy: {acc_str}")


def _normalize(value, feature_name):
    """Normalize a single feature value to [0, 1] using calibrated min/max."""
    mn = _feature_min[feature_name]
    mx = _feature_max[feature_name]
    if mx == mn:
        return 0.0
    return max(0.0, min(1.0, (float(value) - mn) / (mx - mn)))


def compute_biomarkers(nutrition):
    """
    Compute 6 universal biomarker scores from a food's nutritional data.
    
    Each biomarker is a weighted combination of normalized nutritional features,
    producing a score in [0, 1] where higher = more risky.
    
    Args:
        nutrition: A dict-like object (e.g., pandas Series) with FEATURE_NAMES as keys.
    
    Returns:
        dict mapping biomarker names to scores in [0.0, 1.0].
    """
    if not _calibrated:
        raise RuntimeError("Biomarker engine not calibrated. Call calibrate(df) first.")

    # Normalize each raw feature to [0, 1]
    n = {feat: _normalize(nutrition[feat], feat) for feat in FEATURE_NAMES}

    biomarkers = {
        # High sugar + high carbs - fibre dampening = blood sugar spike
        'glycemic_impact': (
            0.55 * n['Free Sugar'] + 0.30 * n['Carbohydrate'] - 0.15 * n['Fibre']
        ),
        # High sodium + high fats = cardiovascular load
        'cardiovascular_strain': (
            0.55 * n['Sodium'] + 0.35 * n['Fats'] + 0.10 * n['Calories']
        ),
        # High fats + high calories = cholesterol risk
        'lipid_load': (
            0.60 * n['Fats'] + 0.30 * n['Calories'] - 0.10 * n['Fibre']
        ),
        # High calories + low fibre + low vitamin C = inflammation
        'inflammatory_index': (
            0.35 * n['Calories'] + 0.30 * (1.0 - n['Fibre'])
            + 0.20 * (1.0 - n['Vitamin C']) + 0.15 * n['Free Sugar']
        ),
        # Low protein + low iron + low calcium = thyroid strain
        'thyroid_stress': (
            0.35 * (1.0 - n['Protein']) + 0.30 * (1.0 - n['Iron'])
            + 0.25 * (1.0 - n['Calcium']) + 0.10 * (1.0 - n['Folate'])
        ),
        # High calories + high fats = weight gain
        'caloric_density': (
            0.80 * n['Calories'] + 0.20 * n['Fats']
        ),
    }

    # Clamp all values to [0.0, 1.0]
    return {k: max(0.0, min(1.0, v)) for k, v in biomarkers.items()}


def evaluate_food_for_disease(biomarkers, disease, food_nutrition=None):
    """
    Evaluate whether a food is suitable for a given disease.
    
    Args:
        biomarkers: dict from compute_biomarkers().
        disease: disease key string (e.g., 'diabetes').
        food_nutrition: optional Series/dict with raw nutrition values for richer explanations.
    
    Returns:
        (verdict, explanation_markdown) tuple.
        verdict: "GOOD TO EAT" or "AVOID".
        explanation_markdown: Human-readable markdown explanation.
    """
    if disease not in DISEASE_PROFILES:
        return "UNKNOWN", f"Disease '{disease}' is not currently supported."

    profile = DISEASE_PROFILES[disease]
    weights = profile['weights']
    threshold = profile['threshold']

    # Compute weighted risk score
    risk_score = sum(biomarkers[bm] * w for bm, w in weights.items())

    verdict = "GOOD TO EAT" if risk_score <= threshold else "AVOID"

    # Build explanation — list each biomarker with its score and risk level
    parts = []
    for bm_name, weight in sorted(weights.items(), key=lambda x: -biomarkers[x[0]]):
        score = biomarkers[bm_name]
        info = BIOMARKER_INFO[bm_name]

        if score > 0.60:
            level = "🔴 High"
        elif score > 0.35:
            level = "🟡 Moderate"
        else:
            level = "🟢 Low"

        parts.append(f"- {info['display']}: **{score:.2f}** ({level})")

    explanation = "\n".join(parts)

    # Append key nutrient values if raw nutrition data is provided
    if food_nutrition is not None:
        nutrient_parts = []
        seen = set()
        for bm_name in weights:
            for nutrient in BIOMARKER_INFO[bm_name]['key_nutrients']:
                if nutrient not in seen:
                    seen.add(nutrient)
                    val = float(food_nutrition[nutrient])
                    unit = NUTRIENT_UNITS.get(nutrient, '')
                    nutrient_parts.append(f"{nutrient}: {val:.1f}{unit}")

        if nutrient_parts:
            explanation += f"\n\n_Key nutrients: {' | '.join(nutrient_parts)}_"

    return verdict, explanation


def get_supported_diseases():
    """Return a list of all supported disease keys."""
    return list(DISEASE_PROFILES.keys())


def get_disease_display_name(disease):
    """Return the human-friendly display name for a disease key."""
    if disease in DISEASE_PROFILES:
        return DISEASE_PROFILES[disease]['display']
    return disease.replace('_', ' ').title()
