
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("dataset/career_dataset.csv")
X = df.drop("Career", axis=1)
y = df["Career"]

model = joblib.load("saved_models/career_model.pkl")
pred = model.predict(X)

print("Accuracy:", accuracy_score(y, pred))
print(classification_report(y, pred))
print(confusion_matrix(y, pred))
