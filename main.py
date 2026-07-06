from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from chatbot import get_chatbot_response, reset_conversation
from biomarker_engine import calibrate, compute_biomarkers, evaluate_food_for_disease, get_supported_diseases, get_disease_display_name

app = Flask(__name__)
CORS(app)

# --- Load Data and Calibrate Biomarker Engine at Startup ---
try:
    df = pd.read_csv('food_nutrition_data.csv')
    df.dropna(subset=['Dish Name'], inplace=True)
    df.set_index('Dish Name', inplace=True)
    df.index = df.index.str.lower()
    df = df[~df.index.duplicated(keep='first')]

    # Calibrate the biomarker engine with the dataset
    calibrate(df)

    print("Data loaded and biomarker engine calibrated successfully.")
except FileNotFoundError:
    print("\n--- FATAL ERROR ---")
    print("food_nutrition_data.csv not found. Please make sure the file is in the correct folder.")
    df = None

# Optionally load XGBoost models (kept for reference/comparison)
try:
    models = joblib.load('nutrition_models.pkl')
    print("XGBoost models also loaded (available as fallback).")
except FileNotFoundError:
    models = None
    print("Note: nutrition_models.pkl not found. Using biomarker engine only.")


@app.route('/chat', methods=['POST'])
def chat():
    if df is None:
        return jsonify({"error": "Data not loaded. Please ensure food_nutrition_data.csv is present."}), 500

    data = request.json
    user_message = data.get('message', '')

    response_text, extracted_info = get_chatbot_response(user_message)

    if extracted_info:
        food = extracted_info['food_item']
        conditions = extracted_info['conditions']

        analysis_parts = []

        try:
            # Lookup food in the dataset (index is lowercase)
            food_row = df.loc[food.lower()]

            # Compute biomarkers once for this food
            biomarkers = compute_biomarkers(food_row)

            for condition in conditions:
                if condition in get_supported_diseases():
                    verdict, explanation = evaluate_food_for_disease(
                        biomarkers, condition, food_row
                    )
                    display_name = get_disease_display_name(condition)
                    analysis_parts.append(
                        f"**For {display_name}: {verdict}**\n{explanation}"
                    )
                else:
                    analysis_parts.append(
                        f"**For {condition.replace('_', ' ').title()}:** "
                        f"_Not yet supported. Coming soon!_"
                    )

            # Assemble the final response
            if analysis_parts:
                full_analysis = "\n\n".join(analysis_parts)
                final_response = (
                    f"{response_text}\n\n"
                    f"### Nutritional Analysis for {food.title()}\n"
                    f"{full_analysis}"
                    f"\n\n*Disclaimer: This is AI-generated advice based on nutritional "
                    f"biomarkers. Please consult a healthcare professional.*"
                )
            else:
                final_response = (
                    "I couldn't analyze the conditions you mentioned. Please try again."
                )

        except KeyError:
            final_response = (
                f"Sorry, I couldn't find the nutritional details for "
                f"'{food.title()}' in my database."
            )

        reset_conversation()
        return jsonify({'response': final_response})
    else:
        return jsonify({'response': response_text})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
