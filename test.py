import pandas as pd
import joblib
from custom_transformers import MultiColumnTransformer

# Load model
model = joblib.load("career_recommendation_model.pkl")

# Create test samples
test_samples = pd.DataFrame([
    {
        "Age": 22,
        "Education": "Bachelor",
        "Skills": ["Python", "Data Analysis", "Statistics"],
        "Personality_Traits": ["Analytical", "Detail Oriented"],
        "Years_of_Experience": 0
    },
    {
        "Age": 28,
        "Education": "Master",
        "Skills": ["Machine Learning", "Python", "Deep Learning"],
        "Personality_Traits": ["Problem Solving", "Innovative"],
        "Years_of_Experience": 4
    },
    {
        "Age": 35,
        "Education": "Bachelor",
        "Skills": ["Project Management", "Agile", "Communication"],
        "Personality_Traits": ["Leadership", "Organized"],
        "Years_of_Experience": 10
    },
    {
        "Age": 24,
        "Education": "Bachelor",
        "Skills": ["HTML", "CSS", "JavaScript"],
        "Personality_Traits": ["Creative", "Detail Oriented"],
        "Years_of_Experience": 1
    },
    {
        "Age": 31,
        "Education": "Master",
        "Skills": ["SQL", "Power BI", "Data Visualization"],
        "Personality_Traits": ["Analytical", "Critical Thinking"],
        "Years_of_Experience": 7
    },
    {
        "Age": 26,
        "Education": "Bachelor",
        "Skills": ["Networking", "Linux", "Security"],
        "Personality_Traits": ["Responsible", "Problem Solving"],
        "Years_of_Experience": 3
    },
    {
        "Age": 29,
        "Education": "Master",
        "Skills": ["Cloud Computing", "AWS", "Docker"],
        "Personality_Traits": ["Adaptable", "Technical"],
        "Years_of_Experience": 5
    },
    {
        "Age": 23,
        "Education": "Bachelor",
        "Skills": ["Java", "OOP", "Data Structures"],
        "Personality_Traits": ["Logical", "Persistent"],
        "Years_of_Experience": 1
    },
    {
        "Age": 34,
        "Education": "PhD",
        "Skills": ["Research", "Statistics", "Python"],
        "Personality_Traits": ["Curious", "Analytical"],
        "Years_of_Experience": 8
    },
    {
        "Age": 27,
        "Education": "Bachelor",
        "Skills": ["Digital Marketing", "SEO", "Content Strategy"],
        "Personality_Traits": ["Creative", "Communicative"],
        "Years_of_Experience": 4
    }
])

# Predict
predictions = model.predict(test_samples)

# Show results
for i, career in enumerate(predictions):
    print(f"Sample {i+1}: Recommended Career → {career}")
