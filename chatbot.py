import pandas as pd
from thefuzz import process, fuzz

# This chatbot is upgraded to handle multiple health conditions,
# with improved fuzzy matching, greeting detection, and confirmation flow.

# --- Data Loading & Synonym Mapping ---
try:
    df = pd.read_csv('food_nutrition_data.csv')
    KNOWN_FOODS = list(df['Dish Name'].str.lower().unique())

    ALL_CONDITIONS_WITH_ALIASES = {
        "diabetes": [
            "diabetes", "diabetic", "sugar problem", "blood sugar",
            "type 2 diabetes", "type 1 diabetes", "high sugar", "sugar patient",
        ],
        "hypertension": [
            "hypertension", "high blood pressure", "high bp", "bp issue",
            "blood pressure", "heart disease", "heart problem", "bp",
        ],
        "hyperlipide": [
            "hyperlipide", "hyperlipidemia", "cholesterol", "high cholesterol",
            "lipid", "triglycerides", "high triglycerides",
        ],
        "thyroid": [
            "thyroid", "thyroid disorder", "hypothyroid", "hyperthyroid",
            "thyroid problem",
        ],
        "obesity": [
            "obesity", "obese", "overweight", "weight issue",
            "weight loss", "weight management",
        ],
        "kidney_disease": [
            "kidney disease", "kidney problem", "renal disease",
            "kidney", "renal",
        ],
        "pcod": [
            "pcod", "pcos", "polycystic", "polycystic ovary",
        ],
    }

    ALIAS_TO_CANONICAL = {
        alias: canonical
        for canonical, aliases in ALL_CONDITIONS_WITH_ALIASES.items()
        for alias in aliases
    }
    ALL_ALIASES = list(ALIAS_TO_CANONICAL.keys())
    print("Chatbot loaded known foods and expanded condition aliases.")

except FileNotFoundError:
    print("WARNING: food_nutrition_data.csv not found.")
    KNOWN_FOODS = []
    ALL_ALIASES = []
    ALIAS_TO_CANONICAL = {}

# --- Greeting, Thanks & Confirmation Patterns ---
GREETINGS = {
    "hi", "hello", "hey", "hola", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
    "what's up", "whats up", "sup",
}
THANKS = {"thanks", "thank you", "thankyou", "thx", "ty", "appreciate"}
AFFIRM = {
    "yes", "yeah", "yep", "yup", "correct", "right",
    "sure", "ok", "okay", "y", "yea", "absolutely", "definitely",
}
DENY = {"no", "nah", "nope", "wrong", "n", "negative"}

# --- State Management (supports confirmation flow) ---
conversation_state = {
    "food_item": None,
    "conditions": [],
    "pending_food": None,  # Food awaiting user confirmation
}


def reset_conversation():
    """Clears the chatbot's memory for a new query."""
    global conversation_state
    conversation_state = {"food_item": None, "conditions": [], "pending_food": None}
    print("Conversation state reset.")


def get_chatbot_response(user_message):
    """
    Processes a user's message, identifies a food and MULTIPLE conditions.
    Now includes greeting detection, confirmation flow, and improved matching.
    """
    global conversation_state
    user_message_clean = user_message.lower().strip()
    words = user_message_clean.split()

    if not words:
        return "I didn't catch that. Could you tell me a food item and your health condition?", None

    # --- 0. Handle Pending Confirmation ---
    if conversation_state['pending_food']:
        first_word = words[0]
        if first_word in AFFIRM or user_message_clean in AFFIRM:
            conversation_state['food_item'] = conversation_state['pending_food']
            conversation_state['pending_food'] = None
            # Fall through to also extract conditions from this message
        elif first_word in DENY or user_message_clean in DENY:
            conversation_state['pending_food'] = None
            return "No problem! Could you tell me the exact food item you're thinking of?", None
        else:
            # Not a yes/no — treat as fresh input, clear pending
            conversation_state['pending_food'] = None

    # --- 1. Intelligent Entity Extraction ---
    CONFIDENCE_HIGH = 80
    CONFIDENCE_LOW = 65

    # Extract food (one per query) — using WRatio for robust multi-word and substring matching
    if not conversation_state['food_item']:
        best_food_match = process.extractOne(
            user_message_clean, KNOWN_FOODS, scorer=fuzz.WRatio
        )
        if best_food_match:
            if best_food_match[1] >= CONFIDENCE_HIGH:
                conversation_state['food_item'] = best_food_match[0]
            elif best_food_match[1] >= CONFIDENCE_LOW:
                # Uncertain match — ask for confirmation
                conversation_state['pending_food'] = best_food_match[0]

    # Extract ALL matching conditions
    best_condition_matches = process.extract(
        user_message_clean, ALL_ALIASES, scorer=fuzz.partial_ratio
    )
    for match, score in best_condition_matches:
        if score > CONFIDENCE_HIGH:
            canonical_condition = ALIAS_TO_CANONICAL.get(match)
            if canonical_condition and canonical_condition not in conversation_state['conditions']:
                conversation_state['conditions'].append(canonical_condition)

    # --- 2. Conversational Logic ---
    food = conversation_state['food_item']
    conditions = conversation_state['conditions']
    pending = conversation_state['pending_food']

    # If a food is pending confirmation, ask before proceeding
    if pending and not food:
        return f"Did you mean **{pending.title()}**?", None

    if food and conditions:
        # SUCCESS: We have everything we need.
        conditions_str = " and ".join([c.replace('_', ' ').capitalize() for c in conditions])
        response_text = f"Analyzing **{food.title()}** for **{conditions_str}**... Here's what I found:"
        extracted_info = {
            'food_item': conversation_state['food_item'],
            'conditions': list(conversation_state['conditions']),
        }
        return response_text, extracted_info

    elif food and not conditions:
        return (
            f"Got it, you're asking about **{food.title()}**. "
            f"What health condition(s) should I check for? "
            f"_(e.g., diabetes, high bp, cholesterol, obesity)_"
        ), None

    elif conditions and not food:
        conditions_str = ", ".join([c.replace('_', ' ').capitalize() for c in conditions])
        return (
            f"Noted, checking for **{conditions_str}**. "
            f"What food item are you curious about?"
        ), None

    else:
        # Nothing extracted — check for greetings or thanks
        is_greeting = len(words) <= 4 and any(w in GREETINGS for w in words[:2])
        is_thanks = any(t in user_message_clean for t in THANKS)

        if is_greeting:
            return (
                "Hello! 👋 I'm your AI Nutritionist. Tell me a food item and your "
                "health condition, and I'll analyze if it's safe to eat!\n\n"
                "For example, try: *\"Can I eat rice if I have diabetes?\"*"
            ), None
        elif is_thanks:
            return "You're welcome! 😊 Feel free to ask me about any food anytime.", None
        else:
            return (
                "I'm your AI Nutritionist! 🥗 Tell me a food and your health condition, "
                "and I'll check if it's safe for you.\n\n"
                "For example: *\"Is Idli good for high blood pressure?\"*"
            ), None
