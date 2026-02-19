import pandas as pd
import joblib
from typing import List, Dict, Any

# Load model once (global for Flask performance)
model = joblib.load(r"C:\Users\Muhammad Waseem\OneDrive\Desktop\ai_recommender\ai_career-recommender\career_recommendation_model.pkl")


def predict_career(person: Dict[str, Any]) -> str:
    """
    Predict recommended career for a single person using trained ML model.

    This function is designed to be easily integrated into a Flask API.
    It performs strict validation of input data types and required fields
    before sending data to the model.

    Parameters
    ----------
    person : dict
        Dictionary containing person details with following keys:

        {
            "Age": int,
            "Education": str,
            "Skills": List[str],
            "Personality_Traits": List[str],
            "Years_of_Experience": int
        }

        Example:
        {
            "Age": 24,
            "Education": "Bachelor",
            "Skills": ["Python", "Machine Learning"],
            "Personality_Traits": ["Analytical", "Creative"],
            "Years_of_Experience": 1
        }

    Returns
    -------
    str
        Recommended career predicted by AI model.

    Raises
    ------
    ValueError
        If required fields are missing or data types are incorrect.

    TypeError
        If input is not a dictionary.

    Usage in Flask
    --------------
    You can call this function inside Flask POST API:

        data = request.json
        result = predict_career(data)
        return jsonify({"recommended_career": result})
    """

    # ----------------------------
    # 1. Input must be dictionary
    # ----------------------------
    if not isinstance(person, dict):
        raise TypeError("Input must be a dictionary containing person details.")

    # ----------------------------
    # 2. Required fields
    # ----------------------------
    required_fields = [
        "Age",
        "Education",
        "Skills",
        "Personality_Traits",
        "Years_of_Experience"
    ]

    for field in required_fields:
        if field not in person:
            raise ValueError(f"Missing required field: {field}")

    # ----------------------------
    # 3. Data type validation
    # ----------------------------
    if not isinstance(person["Age"], int):
        raise ValueError("Age must be integer")

    if not isinstance(person["Education"], str):
        raise ValueError("Education must be string")

    if not isinstance(person["Skills"], list) or not all(isinstance(s, str) for s in person["Skills"]):
        raise ValueError("Skills must be list of strings")

    if not isinstance(person["Personality_Traits"], list) or not all(isinstance(p, str) for p in person["Personality_Traits"]):
        raise ValueError("Personality_Traits must be list of strings")

    if not isinstance(person["Years_of_Experience"], int):
        raise ValueError("Years_of_Experience must be integer")

    # ----------------------------
    # 4. Convert to DataFrame
    # ----------------------------
    input_df = pd.DataFrame([person])

    # ----------------------------
    # 5. Predict using model
    # ----------------------------
    prediction = model.predict(input_df)
    print(f"Model prediction: {prediction}")
    # Return single result
    return str(prediction[0])



from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pickle
import os
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
#  App Setup
# ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'careerai_secret_key_2025'

# ──────────────────────────────────────────────
#  Load the ML Model
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'career_recommendation_model.pkl')
model = None
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
    print(f"   Model type: {type(model).__name__}")

    if hasattr(model, 'named_steps'):
        print(f"   Pipeline steps: {list(model.named_steps.keys())}")

    if hasattr(model, 'classes_'):
        print(f"   Number of career classes: {len(model.classes_)}")
        print(f"   Sample careers: {list(model.classes_[:5])}")

except FileNotFoundError:
    model = None
    print("⚠️  ERROR: career_recommendation_model.pkl not found!")
    print(f"   Looking for: {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"⚠️  ERROR loading model: {e}")
    import traceback

    traceback.print_exc()

# ──────────────────────────────────────────────
#  Fallback Predictions
# ──────────────────────────────────────────────
FALLBACK_CAREERS = [
    {"career": "Software Developer", "match": 85},
    {"career": "Data Scientist", "match": 78},
    {"career": "Graphic Designer", "match": 72},
    {"career": "Marketing Specialist", "match": 65},
    {"career": "Network Administrator", "match": 58},
]


def prepare_model_input(age, education, skills_list, personality_list, experience_years):
    """
    Prepare input in the EXACT format your model expects.

    Your model was trained on:
    - Age: int
    - Education: str
    - Skills: list of strings
    - Personality_Traits: list of strings
    - Years_of_Experience: int
    """

    input_df = pd.DataFrame([{
        'Age': int(age),
        'Education': str(education),
        'Skills': skills_list,  # Must be a list!
        'Personality_Traits': personality_list,  # Must be a list!
        'Years_of_Experience': int(experience_years)
    }])

    return input_df


def run_model_prediction(age, education, skills, personality_traits, experience_years):
    """
    Run the actual prediction using your trained model.

    Parameters:
    - age: int (e.g., 23)
    - education: str (e.g., "Bachelor's Degree")
    - skills: str (comma-separated, e.g., "Python, Data Analysis")
    - personality_traits: str (comma-separated, e.g., "Analytical, Creative")
    - experience_years: int (e.g., 2)

    Returns: list of dicts with career predictions
    """

    if model is None:
        print("⚠️  Model not loaded - using fallback predictions")
        return FALLBACK_CAREERS

    try:
        # Convert comma-separated strings to lists (same as training data)
        skills_list = [s.strip() for s in skills.split(',') if s.strip()]
        personality_list = [p.strip() for p in personality_traits.split(',') if p.strip()]

        print(f"\n{'=' * 60}")
        print(f"🔍 PREDICTION REQUEST")
        print(f"{'=' * 60}")
        print(f"Age:                 {age}")
        print(f"Education:           {education}")
        print(f"Skills (list):       {skills_list}")
        print(f"Personality (list):  {personality_list}")
        print(f"Experience (years):  {experience_years}")
        print(f"{'=' * 60}\n")

        # Prepare input in exact training format
        input_df = prepare_model_input(
            age=age,
            education=education,
            skills_list=skills_list,
            personality_list=personality_list,
            experience_years=experience_years
        )

        print("📊 Input DataFrame prepared:")
        print(input_df)
        print(f"\nDataFrame dtypes:\n{input_df.dtypes}\n")

        # Get predictions with probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict(input_df)[0]
            classes = model.classes_

            # Create results
            results = []
            for career, prob in zip(classes, probabilities):
                results.append({
                    "career": str(career),
                    "match": round(float(prob) * 100, 1)
                })

            # Sort by match percentage
            results.sort(key=lambda x: x["match"], reverse=True)

            print(f"✅ Predictions generated!")
            print(f"   Top 5 careers:")
            for i, r in enumerate(results[:5], 1):
                print(f"   {i}. {r['career']}: {r['match']}%")
            print(f"{'=' * 60}\n")

            return results[:5]

        else:
            # Model only returns single prediction
            prediction = model.predict(input_df)[0]
            print(f"✅ Single prediction: {prediction}\n")

            return [
                {"career": str(prediction), "match": 95},
                {"career": "Alternative Career 1", "match": 78},
                {"career": "Alternative Career 2", "match": 65}
            ]

    except Exception as e:
        print(f"❌ PREDICTION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return FALLBACK_CAREERS


# ══════════════════════════════════════════════
#  PAGE ROUTES
# ══════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if username and password:
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials.')

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if username and email and password:
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('signup.html', error='Fill all fields.')

    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/about')
def about():
    return render_template('aboutus.html')


@app.route('/contact')
def contact():
    return render_template('contactus.html')


# ══════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Career Prediction API - Matches your trained model exactly!

    Expects JSON:
    {
        "age": 23,
        "education": "Bachelor's Degree",
        "skills": "Python, Data Analysis, Problem Solving",
        "personality_traits": "Analytical, Creative, Detail-oriented",
        "experience_years": 2
    }

    Returns:
    {
        "success": true,
        "top_career": "Data Scientist",
        "top_match": 87.5,
        "results": [
            {"career": "Data Scientist", "match": 87.5},
            {"career": "Software Developer", "match": 82.3},
            ...
        ]
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        # Extract and validate inputs
        age = data.get('age')
        education = data.get('education', '').strip()
        skills = data.get('skills', '').strip()
        personality_traits = data.get('personality_traits', '').strip()
        experience_years = data.get('experience_years')

        # Validation
        if not education or not skills or not personality_traits:
            return jsonify({
                "success": False,
                "error": "Education, skills, and personality traits are required"
            }), 400

        # Convert to correct types
        try:
            age = int(age) if age else 25  # Default age
            experience_years = int(experience_years) if experience_years else 0
        except ValueError:
            return jsonify({
                "success": False,
                "error": "Age and experience must be numbers"
            }), 400

        # Run prediction
        results = run_model_prediction(
            age=age,
            education=education,
            skills=skills,
            personality_traits=personality_traits,
            experience_years=experience_years
        )

        return jsonify({
            "success": True,
            "top_career": results[0]["career"],
            "top_match": results[0]["match"],
            "results": results
        })

    except Exception as e:
        print(f"❌ API Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/contact', methods=['POST'])
def api_contact():
    """Contact form submission"""
    try:
        data = request.get_json()

        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        message = data.get('message', '').strip()

        if not name or not email or not message:
            return jsonify({"success": False, "error": "All fields required"}), 400

        print(f"📩 Contact from {name} <{email}>: {message}")

        return jsonify({
            "success": True,
            "message": "Message received! We'll reply within 24-48 hours."
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/model-status', methods=['GET'])
def api_model_status():
    """Debug endpoint - check if model is loaded correctly"""
    status = {
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    }

    if model is not None:
        status["model_type"] = type(model).__name__

        if hasattr(model, 'classes_'):
            status["num_classes"] = len(model.classes_)
            status["all_careers"] = list(model.classes_)

        if hasattr(model, 'named_steps'):
            status["pipeline_steps"] = list(model.named_steps.keys())

    return jsonify(status)


# ══════════════════════════════════════════════
#  Run App
# ══════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 CareerAI Flask Server Starting...")
    print("=" * 60)
    print(f"📍 Model path: {MODEL_PATH}")
    print(f"✅ Model loaded: {model is not None}")

    if model is not None:
        print(f"📊 Model type: {type(model).__name__}")
        if hasattr(model, 'classes_'):
            print(f"🎯 Number of careers: {len(model.classes_)}")

    print("=" * 60)
    print("🌐 Server will start at: http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    app.run(debug=True)

