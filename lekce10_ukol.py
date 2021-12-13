import pandas
import requests
import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC

r = requests.get(
    "https://raw.githubusercontent.com/lutydlitatova/czechitas-datasets/main/datasets/soybean-2-rot.csv")
open("soybean-2-rot.csv", "wb").write(r.content)
r = requests.get(
    "https://raw.githubusercontent.com/lutydlitatova/czechitas-datasets/main/datasets/auto.csv")
open("auto.csv", "wb").write(r.content)

#feature importance:
data = pandas.read_csv("soybean-2-rot.csv")

X = data.drop(columns=["class"])
y = data["class"]

oh_encoder = OneHotEncoder()
X = oh_encoder.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1)
clf.fit(X_train, y_train)
for name, imp in zip(oh_encoder.get_feature_names_out(), clf.feature_importances_):
    print(name,imp)

# největší "důležitost" má proměnná "plant-stand-lt-normal"

y_pred = clf.predict(X_test)
print(f1_score(y_test, y_pred, average="weighted"))
# #print(f1_score(y_test, y_pred, labels=[0.52350127], average="weighted"))

#Auto:
df = pandas.read_csv("auto.csv", na_values=["?"])
df = df.dropna()
avg_mpg = pandas.pivot_table(df, index="year", columns="origin", aggfunc=numpy.mean, values="mpg")
avg_mpg.plot.bar()
plt.show()

X = df.drop(columns=["origin"])
y = df["origin"]

encoder = OneHotEncoder()
X = encoder.fit_transform(X)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

model = DecisionTreeClassifier(random_state=0)
clf = GridSearchCV(model, param_grid={"max_depth": [1, 2, 3], "min_samples_leaf": [1, 3, 5]}, scoring="f1_weighted")
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_score_)
sc_grid = clf.best_score_

y_pred = clf.best_estimator_.predict(X_test)
print(round(f1_score(y_test, y_pred, average="weighted"), 2))
# #f1_score: 0.63

# #dobrovolný doplněk:
model_2 = LinearSVC()
clf = GridSearchCV(model_2, param_grid={"C": [0.001, 0.01, 0.1, 1.0]}, scoring="f1_weighted")
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_score_)
sc_lsvc = clf.best_score_

y_pred = clf.best_estimator_.predict(X_test)
print(round(f1_score(y_test, y_pred, average="weighted"), 2))

plt.bar(["GridSearchCV", "LinearSVC"], [sc_grid, sc_lsvc])
plt.show()