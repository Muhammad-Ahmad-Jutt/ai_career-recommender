# career_model_train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

from custom_transformers import MultiColumnTransformer

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv(
    r"C:\Users\Muhammad Waseem\OneDrive\Desktop\ai_recommender\ai_career-recommender\synthetic_created_data.csv"
)

# -------------------------------
# Preprocess Features
# -------------------------------
df['Skills'] = df['Skills'].apply(lambda x: [s.strip() for s in x.split(',')])
df['Personality_Traits'] = df['Personality_Traits'].apply(lambda x: [p.strip() for p in x.split(',')])

X = df[['Age', 'Education', 'Skills', 'Personality_Traits', 'Years_of_Experience']]
y = df['Recommended_Career']

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Pipeline
# -------------------------------
pipeline = Pipeline([
    ('preprocess', ColumnTransformer(
        transformers=[
            ('education', OneHotEncoder(handle_unknown='ignore'), ['Education']),
            (
                'skills_personality',
                MultiColumnTransformer(['Skills', 'Personality_Traits']),
                ['Skills', 'Personality_Traits']
            )
        ],
        remainder='passthrough'
    )),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

# -------------------------------
# Train
# -------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Save
# -------------------------------
joblib.dump(pipeline, "career_recommendation_model.pkl")
print("Model saved successfully")
