import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

df = pd.read_csv("dataset/career_dataset.csv")

X = df.drop("Career", axis=1)
y = df["Career"]

model = joblib.load("saved_models/career_model.pkl")

pred = model.predict(X)

cm = confusion_matrix(y, pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
