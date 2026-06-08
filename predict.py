
import joblib, pandas as pd
model = joblib.load("../saved_models/career_model.pkl")
student = pd.DataFrame([{
"Programming":8,"Mathematics":9,"Communication":6,"ProblemSolving":9,
"AnalyticalThinking":8,"ProjectExperience":7,"Teamwork":6,"Leadership":5,
"Certifications":4,"WebDevelopment":4,"MachineLearning":9,
"CloudComputing":4,"CyberSecurity":3,"DatabaseSkills":5,"CGPA":8
}])
print(model.predict(student)[0])
