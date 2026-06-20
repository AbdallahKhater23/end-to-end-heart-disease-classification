import numpy as np  # np is short for numpy

import pandas as pd  # pandas is so commonly used, it's shortened to pd

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

## Model evaluators
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay

import datetime

print(f"Notebook last updated: {datetime.datetime.now()}\n")

df = pd.read_csv("heart_disease_classification_dataset.csv")


df = df.dropna()
df["target"].value_counts(normalize=True)

print(df["sex"].unique())
df["sex"] = df["sex"].map({"male": 1, "female": 0})
print(df.head())

df["target"].value_counts().plot(kind="bar", color=["r", "b"])

pd.crosstab(index=df.target, columns=df.sex).plot(
    kind="bar", figsize=(10, 6), color=["r", "b"]
)
plt.title("Heart Disease Frequency vs Sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female, Male"])
plt.xticks(rotation=0)

print(df["target"].unique())
print(df["target"].dtype)
df["target"] = df["target"].map({"yes": 1, "no": 0})
df["target"] = df["target"].astype(int)

plt.figure(figsize=(10, 6))

plt.scatter(df.age[df.target == 1], df.thalach[df.target == 1], c="salmon")
plt.scatter(df.age[df.target == 0], df.thalach[df.target == 0], c="lightblue")
plt.title("Heart disease in fuction of age and heart rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "NO Disease"])

df.age.plot.hist()
pd.crosstab(index=df.cp, columns=df.target).plot(
    kind="bar", figsize=(10, 6), color=["r", "b"]
)

plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("chest pain type")
plt.ylabel("frequency")
plt.legend(["No Disease", "Disease"])
plt.xticks(rotation=0)

corr_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
# plt.show()

df.hist(column="target", figsize=(8, 5))
# plt.show()

X = df.drop("target", axis=1)
y = df["target"]

np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=100),
    "Random Forest": RandomForestClassifier(),
}


def fit_and_score(models, X_train, y_train, X_test, y_test):
    np.random.seed(42)

    model_scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


model_scores = fit_and_score(
    models=models, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)
model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar()

train_scores = []
test_scores = []
neighbors = range(1, 21)
knn = KNeighborsClassifier()

for i in neighbors:
    knn.set_params(n_neighbors=i)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


log_reg_grid = {"C": np.logspace(-4, 4, 20), "solver": ["liblinear"]}

rf_grid = {
    "n_estimators": np.arange(10, 1000, 50),
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 20, 2),
}

# %%time

np.random.seed(42)

rs_log_reg = RandomizedSearchCV(
    LogisticRegression(),
    param_distributions=log_reg_grid,
    cv=5,
    n_iter=20,
    verbose=True,
)

rs_log_reg.fit(X_train, y_train)
rs_log_reg.best_params_
rs_log_reg.score(X_test, y_test)

# %%time

np.random.seed(42)

rs_rf = RandomizedSearchCV(
    RandomForestClassifier(), param_distributions=rf_grid, cv=5, n_iter=20, verbose=True
)

rs_rf.fit(X_train, y_train)
rs_rf.best_params_
rs_rf.score(X_test, y_test)

# %%time

log_reg_grid = {"C": np.logspace(-4, 4, 20), "solver": ["liblinear"]}

gs_log_reg = GridSearchCV(
    LogisticRegression(), param_grid=log_reg_grid, cv=5, verbose=True
)

gs_log_reg.fit(X_train, y_train)
gs_log_reg.best_params_
gs_log_reg.score(X_test, y_test)

y_preds = gs_log_reg.predict(X_test)

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(estimator=gs_log_reg, X=X_test, y=y_test)
print(confusion_matrix(y_test, y_preds))

sns.set(font_scale=1.5)


def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")


plot_conf_mat(y_test, y_preds)

print(classification_report(y_test, y_preds))

gs_log_reg.best_params_
clf = LogisticRegression(C=0.23357214690901212, solver="liblinear")

# %%time

cv_acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
cv_acc
cv_acc = np.mean(cv_acc)
cv_acc

cv_precision = np.mean(cross_val_score(clf, X, y, cv=5, scoring="precision"))
cv_precision

cv_recall = np.mean(cross_val_score(clf, X, y, cv=5, scoring="recall"))
cv_recall

cv_f1 = np.mean(cross_val_score(clf, X, y, cv=5, scoring="f1"))
cv_f1

cv_metrics = pd.DataFrame(
    {"Accuracy": cv_acc, "Precision": cv_precision, "Recall": cv_recall, "F1": cv_f1},
    index=[0],
)
cv_metrics.T.plot.bar(title="Cross-Validated Metrics", legend=False)
clf.fit(X_train, y_train)
clf.coef_
features_dict = dict(zip(df.columns, list(clf.coef_[0])))
features_dict
features_df = pd.DataFrame(features_dict, index=[0])
features_df.T.plot.bar(title="Feature Importance", legend=False)
pd.crosstab(df["sex"], df["target"])
pd.crosstab(df["slope"], df["target"])
