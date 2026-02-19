"""
DEBUG SCRIPT - Test your model directly without Flask
Run this to see EXACTLY what the model is doing
"""

import pickle
import pandas as pd
import numpy as np

# Load the model
print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

with open('career_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"✅ Model loaded: {type(model).__name__}")
print(f"✅ Model classes: {list(model.classes_)}\n")

# Test Case 1: Software Developer profile
print("=" * 60)
print("TEST 1: SOFTWARE DEVELOPER PROFILE")
print("=" * 60)

input1 = pd.DataFrame([{
    'Age': 23,
    'Education': "Bachelor's Degree",
    'Skills': ['Python', 'Java', 'Programming', 'Problem Solving'],
    'Personality_Traits': ['Analytical', 'Logical', 'Detail-oriented'],
    'Years_of_Experience': 2
}])

print("Input DataFrame:")
print(input1)
print(f"\nSkills type: {type(input1['Skills'].iloc[0])}")
print(f"Skills value: {input1['Skills'].iloc[0]}")
print(f"Personality type: {type(input1['Personality_Traits'].iloc[0])}")
print(f"Personality value: {input1['Personality_Traits'].iloc[0]}")

try:
    probs1 = model.predict_proba(input1)[0]
    top_indices = np.argsort(probs1)[::-1][:5]

    print("\n✅ Prediction successful!")
    print("Top 5 careers:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. {model.classes_[idx]}: {probs1[idx] * 100:.1f}%")
except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    import traceback

    traceback.print_exc()

# Test Case 2: Graphic Designer profile
print("\n" + "=" * 60)
print("TEST 2: GRAPHIC DESIGNER PROFILE")
print("=" * 60)

input2 = pd.DataFrame([{
    'Age': 25,
    'Education': "Bachelor's Degree",
    'Skills': ['Design', 'Photoshop', 'Illustrator', 'Creativity'],
    'Personality_Traits': ['Creative', 'Artistic', 'Visual'],
    'Years_of_Experience': 1
}])

print("Input DataFrame:")
print(input2)

try:
    probs2 = model.predict_proba(input2)[0]
    top_indices = np.argsort(probs2)[::-1][:5]

    print("\n✅ Prediction successful!")
    print("Top 5 careers:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. {model.classes_[idx]}: {probs2[idx] * 100:.1f}%")
except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    import traceback

    traceback.print_exc()

# Test Case 3: Marketing Specialist profile
print("\n" + "=" * 60)
print("TEST 3: MARKETING SPECIALIST PROFILE")
print("=" * 60)

input3 = pd.DataFrame([{
    'Age': 27,
    'Education': "Master's Degree",
    'Skills': ['Marketing', 'Communication', 'Social Media', 'Sales'],
    'Personality_Traits': ['Outgoing', 'Persuasive', 'Social'],
    'Years_of_Experience': 3
}])

print("Input DataFrame:")
print(input3)

try:
    probs3 = model.predict_proba(input3)[0]
    top_indices = np.argsort(probs3)[::-1][:5]

    print("\n✅ Prediction successful!")
    print("Top 5 careers:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. {model.classes_[idx]}: {probs3[idx] * 100:.1f}%")
except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    import traceback

    traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

try:
    test1_top = model.classes_[np.argmax(probs1)]
    test2_top = model.classes_[np.argmax(probs2)]
    test3_top = model.classes_[np.argmax(probs3)]

    print(f"Test 1 top career: {test1_top}")
    print(f"Test 2 top career: {test2_top}")
    print(f"Test 3 top career: {test3_top}")

    if test1_top == test2_top == test3_top:
        print("\n❌ PROBLEM: All tests return the SAME career!")
        print("   The model is not responding to different inputs.")
        print("   Possible causes:")
        print("   1. Model wasn't trained properly")
        print("   2. Feature extraction is broken")
        print("   3. Model is overfitted to one class")
    else:
        print("\n✅ SUCCESS: Model returns DIFFERENT careers for different inputs!")
        print("   Your model is working correctly.")
except:
    print("\n❌ Could not generate summary (predictions failed)")

print("=" * 60)