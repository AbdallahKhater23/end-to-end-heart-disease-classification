import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

df = pd.read_csv("heart_disease_classification_dataset.csv")

df = df.dropna()
df["target"].value_counts(normalize=True)

print(df["sex"].unique())
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["target"] = df["target"].replace({"yes": 1, "no": 0})
print(df.head())


def crosstab(dataframe, feature_column, target_column):
    pd.crosstab(dataframe[feature_column], dataframe[target_column]).plot(
        kind="bar", color=["salmon", "lightblue"]
    )
    plt.title(f"Relationship between {feature_column} and {target_column}")
    plt.title(f"Relationship between {feature_column} and {target_column}")
    plt.ylabel(f"Count of {target_column.title()}")
    plt.xticks(rotation=0)
    plt.show()


# crosstab(df, 'target', 'sex')


# # plt.figure(figsize=(10, 6))

# # plt.scatter(df.age[df.target == 1], df.thalach[df.target == 1], c="salmon")

# # plt.scatter(df.age[df.target == 0], df.thalach[df.target == 0], c="lightblue")

# # plt.title("Age and its relationship with Heart Rate")
# # plt.xlabel("Age")
# # plt.ylabel("Max Heart Rate")
# # plt.legend(["Disease", "No disease"])
# # plt.

# crosstab(df, "cp", "target")


# corr_matrix = df.corr()
# plt.figure(figsize=(15, 10))
# sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")

df = df.drop(columns=["Unnamed: 0"], errors="ignore")

X = df.drop(labels="target", axis=1)
y = df["target"].astype(int).to_numpy()
print(y)
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(),
}


def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


model_scores = fit_and_score(
    models=models, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)
print(model_scores)

# model_compare = pd.DataFrame(model_scores, index=["accuracy"])
# model_compare.T.plot.bar()

train_scores = []

test_scores = []

neighbors = range(1, 21)  # 1 to 20
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
plt.show()
print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")

log_reg_grid = {"C": np.logspace(-4, 4, 20), "solver": ["liblinear"]}

rf_grid = {
    "n_estimators": np.arange(10, 1000, 50),
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 20, 2),
}

np.random.seed(42)

rs_log_reg = RandomizedSearchCV(
    LogisticRegression(),
    param_distributions=log_reg_grid,
    cv=5,
    n_iter=20,
    verbose=True,
)

rs_log_reg.fit(X_train, y_train)
rs_log_reg.score(X_test, y_test)

np.random.seed(42)

rs_rf = RandomizedSearchCV(
    RandomForestClassifier(), param_distributions=rf_grid, cv=5, n_iter=20, verbose=True
)

rs_rf.fit(X_train, y_train)
rs_rf.best_params_
rs_rf.score(X_test, y_test)

log_reg_grid = {"C": np.logspace(-4, 4, 20), "solver": ["liblinear"]}

gs_log_reg = GridSearchCV(
    LogisticRegression(), param_grid=log_reg_grid, cv=5, verbose=True
)

gs_log_reg.fit(X_train, y_train)
gs_log_reg.best_params_
gs_log_reg.score(X_test, y_test)

y_preds = gs_log_reg.predict(X_test)
y_preds
y_test

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(estimator=gs_log_reg, X=X_test, y=y_test)

print(confusion_matrix(y_test, y_preds))

sns.set(font_scale=1.5)


def plot_conf_mat(y_test, y_preds):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, cbar=False)
    plt.xlabel("true label")
    plt.ylabel("predicted label")


plot_conf_mat(y_test, y_preds)

print(classification_report(y_test, y_preds))

gs_log_reg.best_params_
clf = LogisticRegression(C=0.23357214690901212, solver="liblinear")


def get_cv_metrics(model, X, y, cv=5):
    metrics = ["accuracy", "precision", "recall", "f1"]
    cv_metrics = {}

    for metric in metrics:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
        cv_metrics[metric] = np.mean(scores)
    return cv_metrics


cv_results = get_cv_metrics(clf, X, y, cv=5)
print(cv_results)

cv_df = pd.DataFrame(cv_results, index=["score"])
cv_df.T.plot.bar(title="Comparison of CV Metrics", legend=False)
plt.xticks(rotation=0)
plt.show()
