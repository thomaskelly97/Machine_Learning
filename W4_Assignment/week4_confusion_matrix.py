import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as prep
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve


def calculate_confusion_matrix(truth, pred, model):
    # cf = confusion_matrix(truth, pred)
    tn, fp, fn, tp = confusion_matrix(truth, predictions).ravel()
    print("--- Model ", model, " ---")
    print("Confusion Matrix (tn, fp, fn, tp): ", tn, fp, fn, tp)


def plot_roc_curve(y_test, y_score, model):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr)
    plt.title(model + " ROC Curve")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()


df = pd.read_csv("week4_set2.csv")

X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

# --- KNN MODEL ----
knn_model = KNeighborsClassifier(
    n_neighbors=5, weights="uniform")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

knn_model.fit(X_train, y_train)
predictions = knn_model.predict(X_test)

calculate_confusion_matrix(y_test, predictions, "KNN")
y_score = knn_model.predict_proba(X_test)
plot_roc_curve(y_test, y_score[:, 1], "KNN")

# --- LOGISTIC REGRESSION ---
# not using polynomial features as they dont make much of a difference
polynomial_features = prep.PolynomialFeatures(degree=2)  #  use initial q
X = polynomial_features.fit_transform(X)
log_model = LogisticRegression(C=1, penalty="l2", random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
log_model.fit(X_train, y_train)
predictions = log_model.predict(X_test)

calculate_confusion_matrix(
    y_test, predictions, "Logistic regression")
y_score = log_model.decision_function(X_test)
plot_roc_curve(y_test, y_score, "Logistic Regression")

# --- BASELINE ---
dummy_model = DummyClassifier(strategy="most_frequent", random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
dummy_model.fit(X_train, y_train)
predictions = dummy_model.predict(X_test)

calculate_confusion_matrix(
    y_test, predictions, "Dummy Classifier")
y_score = dummy_model.predict_proba(X_test)
plot_roc_curve(y_test, y_score[:, 1], "Baseline Classifier")
