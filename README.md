# Heart Disease Classification Project 🫀

An end-to-end machine learning pipeline that builds, tunes, and evaluates a classification model to predict the presence of heart disease in patients based on clinical attributes.

## Project Overview

This project explores a clinical dataset of ~300 patients to identify key indicators of heart disease. It evaluates multiple machine learning classifiers, optimizes the top performer, and interprets the clinical features driving the model's decisions.

## Key Features

* **Data Cleaning & Transformation:** Handles missing values and encodes categorical features (`sex`, `target`) into machine-readable binary formats.
* **Exploratory Data Analysis (EDA):** Visualizes feature distributions, age vs. heart rate relationships, and correlations using Seaborn heatmaps.
* **Model Evaluation & Competition:** Evaluates and compares K-Nearest Neighbors (KNN), Random Forest Classifier, and Logistic Regression.
* **Hyperparameter Tuning:** Utilizes `RandomizedSearchCV` and `GridSearchCV` to isolate the optimal regularization strength (`C=0.233`, `solver="liblinear"`).
* **Production Validation:** Implements a rigorous 5-fold cross-validation pipeline tracking Accuracy, Precision, Recall, and F1-Score.

## Final Model Performance

The optimized Logistic Regression model achieved a near-perfect evaluation profile on the test set:

* **Overall Accuracy:** 98%
* **Recall (Sensitivity):** 100% (Caught every single positive heart disease case)
* **Precision:** 97%

## Key Clinical Insights

* **Chest Pain (`cp`):** Emerged as the strongest positive indicator of a heart disease diagnosis.
* **ST Slope (`slope`):** Downsloping segments during exercise testing showed an overwhelming correlation with positive diagnoses.
