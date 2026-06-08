
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif

df = pd.read_csv("../dataset/career_dataset.csv")
X = df.drop("Career", axis=1)
y = df["Career"]

selector = SelectKBest(mutual_info_classif, k=9)
selector.fit(X,y)
print(X.columns[selector.get_support()])
