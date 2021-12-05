import sklearn
import requests
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, f1_score, accuracy_score, recall_score
import matplotlib.pyplot as plt

r = requests.get("https://raw.githubusercontent.com/lutydlitatova/czechitas-datasets/main/datasets/kosatce.csv")
open("kosatce.csv", "wb").write(r.content)

#K Nearest Neighbors 1:
data = pandas.read_csv("water-potability.csv")
data = data.dropna()
X = data.drop(columns=["Potability"])
y = data["Potability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
print(precision_score(y_test, y_pred))

ks = [1, 3, 5, 7, 9]
for k in ks:
  clf = KNeighborsClassifier(n_neighbors=k)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print(k, precision_score(y_test, y_pred))
# #nejlépe u metody precision vychází parametr 9, tj. 0,6090909, v lekci se volil parametr 3

clf = KNeighborsClassifier(n_neighbors=9)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(precision_score(y_test, y_pred))
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
# confusion matrix [[188  43]
#                   #[105  67]]
# #$P = \frac{TP}{TP+FP}$, tj. $P = \frac{67}{67+43}$ = 0,60909

#dobrovolný doplněk:
ks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
precision_scores = []
f1_scores = []
accuracy_scores = []
recall_scores = []
for k in ks:
  clf = KNeighborsClassifier(n_neighbors=k)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  precision_scores.append(precision_score(y_test, y_pred))
  f1_scores.append(f1_score(y_test, y_pred))
  accuracy_scores.append(accuracy_score(y_test, y_pred))
  recall_scores.append(recall_score(y_test, y_pred))

plt.plot(ks, precision_scores, f1_scores, accuracy_scores, recall_scores)
plt.legend(["precision", "f1", "accuracy", "recall"])
plt.show()

#K Nearest Neighbors 2:
df = pandas.read_csv("kosatce.csv")
print(df["target"].value_counts(normalize=True))
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
print(f1_score(y_test, y_pred))

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
print(f1_score(y_test, y_pred))
