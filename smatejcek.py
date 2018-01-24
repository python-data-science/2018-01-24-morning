import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data.csv")

df["Gender"] = df["Sex"].map({"male":1, "female":0})
df["Embark_Numeric"] = df["Embarked"].map({"S":1, "C":2, "Q":3})

y_target = df["Survived"]
X_data = df[["Gender", "SibSp", "Fare"]]

model = tree.DecisionTreeClassifier().fit(X_data, y_target) # X and then y
X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size = 0.33, random_state = 42)

print("TRAIN SCORE: {}".format(round(model.score(X_train, y_train), 6)))
# Train Score = 0.919463

print("TEST SCORE: {}".format(round(model.score(X_test, y_test), 6)))
# Test Score = 0.911864

print("\nBraund, Mr. Owen Harris Survied? {}".format(1 in model.predict([[1, 1, 7.25]])))
# Predicts Mr. Braund did not survive; and he did not.

print("R2 Cross Validation Score: {}".format(cross_val_score(model, X_data, y_target, cv = 10, scoring = "r2")))
# Returns: [-0.02857143 -0.02857143  0.00053476 -0.09465241  0.52406417  0.38128342, 0.14331551  0.23850267  0.04812834  0.1372549 ]

print("Accuracy Cross Validation Score: {}".format(cross_val_score(model, X_data, y_target, cv = 10, scoring = "accuracy")))
# Returns: [ 0.75555556  0.75555556  0.7752809  0.74157303  0.88764045  0.85393258  0.78651685  0.82022472  0.7752809  0.79545455]
