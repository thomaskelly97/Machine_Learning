import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy import stats
from sklearn.metrics import accuracy_score


df = pd.read_csv("week2.csv")

X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
y = df.iloc[:, 2]

# (c) (i) Create new features
X3 = X1 * X1
X4 = X2 * X2

X = np.column_stack((X1, X2, X3, X4))
dataframe = pd.DataFrame(
    {'column1': X1, 'column2': X2, 'column4': X3, 'column5': X4, 'column3': y})
minusDF = []
postivieDF = []

for grouped_y, grouping_x in dataframe.groupby(["column3"]):
    if grouped_y == -1:
        minusDF = grouping_x  # X rows that have y = -1
    else:
        positiveDF = grouping_x  #  X rows that have y = +1

for i, C in enumerate([1]):
    # (c) (ii) Train SVM model
    svmModel = LinearSVC(C=C)
    svmModel.fit(X, y)

    predictions = svmModel.predict(X)
    prediction_dataframe = pd.DataFrame(
        {'column1': X1, 'column2': X2, 'column4': X3, 'column5': X4, 'column3': predictions})

    minus_pred_DF = []
    positive_pred_DF = []

    for grouped_y, grouping_x in prediction_dataframe.groupby(["column3"]):
        if grouped_y == -1:
            minus_pred_DF = grouping_x  # X rows that have y = -1
        else:
            positive_pred_DF = grouping_x  #  X rows that have y = +1

    # (c) (iii) Get errors for baseline and svm model
    def baseline_model_error():
        mode_count = stats.mode(y).count  #  Model that predicts mode output
        total_n = len(y)
        error_rate = (total_n - mode_count) / total_n
        return error_rate
    print(baseline_model_error())

    svm_accuracy = accuracy_score(y, predictions)
    svm_error_rate = 1 - svm_accuracy
    print(svm_error_rate)
    plt.figure(1)
    plt.title(
        "Linear SVC Model - 4 Features; C = " + str(C))
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True

    plt.scatter(minusDF.iloc[:, 0], minusDF.iloc[:, 1],
                marker="+", c="black", s=40)
    plt.scatter(positiveDF.iloc[:, 0],
                positiveDF.iloc[:, 1], marker="o", c="black")

    plt.scatter(minus_pred_DF.iloc[:, 0],
                minus_pred_DF.iloc[:, 1], marker="o", c="red", s=6)
    plt.scatter(positive_pred_DF.iloc[:, 0],
                positive_pred_DF.iloc[:, 1], marker="o", c="lime", s=6)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # (c) (iv) Calculate a decision boundary
    decision_boundary_X = np.linspace(-1, 1, 100)
    decision_b_y = (-svmModel.coef_[0][2]/svmModel.coef_[0][1])*(decision_boundary_X**2) - (svmModel.coef_[
        0][0]/svmModel.coef_[0][1])*decision_boundary_X - (svmModel.intercept_/svmModel.coef_[0][1])

    plt.plot(decision_boundary_X, decision_b_y, c="blue")
    plt.legend(["Decision Boundary", "Training data; y = -1", "Training data; y = +1", "Predictions; y = -1", "Predictions; y = +1",
                ], loc=2)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
