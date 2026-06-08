import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("saved_models/career_model.pkl")

features = [
    "Programming",
    "Mathematics",
    "Communication",
    "ProblemSolving",
    "AnalyticalThinking",
    "ProjectExperience",
    "Teamwork",
    "Leadership",
    "Certifications",
    "WebDevelopment",
    "MachineLearning",
    "CloudComputing",
    "CyberSecurity",
    "DatabaseSkills",
    "CGPA"
]

importance = model.feature_importances_

plt.figure(figsize=(10,6))
plt.barh(features, importance)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
