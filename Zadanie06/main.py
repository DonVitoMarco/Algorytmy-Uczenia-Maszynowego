import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from KNN import KNN

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pd.read_csv(path, names=names)

encoder = LabelEncoder()
dataset["class"] = encoder.fit_transform(dataset["class"])

train_set, test_set = train_test_split(dataset, test_size=0.25, random_state=0)

print("Train set size : ", len(train_set))
print("Test set size : ", len(test_set))

knn = KNN(7)
predictions = []
for index, t in test_set.iterrows():
    predictors_only = t[:-1]
    prediction = knn.predict(train_set, predictors_only)
    predictions.append(prediction)

score = knn.evaluate(np.array(test_set.iloc[:, -1]), predictions)
print("KNN Score = ", score)
