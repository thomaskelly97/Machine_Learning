import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("week2.csv")

X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

# (a) (i) Group input data based on y = -1 and y = +1
dataframe = pd.DataFrame({'column1': X1, 'column2': X2, 'column3': y})
minusDF = []
postivieDF = []

for grouped_y, grouping_x in dataframe.groupby(["column3"]):
    if grouped_y == -1:
        minusDF = grouping_x  # X rows that have y = -1
    else:
        positiveDF = grouping_x  #  X rows that have y = +1

# (a) (ii)
logRegressionModel = LogisticRegression()
logRegressionModel.fit(X, y)


# (a) (iii)
predictions = logRegressionModel.predict(X)

prediction_dataframe = pd.DataFrame(
    {'column1': X1, 'column2': X2, 'column3': predictions})
minus_pred_DF = []
positive_pred_DF = []

for grouped_y, grouping_x in prediction_dataframe.groupby(["column3"]):
    if grouped_y == -1:
        minus_pred_DF = grouping_x  # X rows that have y = -1
    else:
        positive_pred_DF = grouping_x  #  X rows that have y = +1

slope = -(logRegressionModel.coef_[0][0]*logRegressionModel.intercept_)/(
    logRegressionModel.intercept_*logRegressionModel.coef_[0][1])
y_intercept = -logRegressionModel.intercept_ / logRegressionModel.coef_[0][1]
decision_boundary_X = np.linspace(-1, 1, 2)
decision_boundary_Y = slope*decision_boundary_X + y_intercept

# PLOT
plt.figure(1)
plt.title(
    "Visualisation of Training Data with Predicted Target Values with Descision Boundary")
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
plt.plot(decision_boundary_X, decision_boundary_Y,
         c="blue", linestyle="dashed")
plt.xlim(-1, 1)
plt.legend(["Decision Boundary", "Training data; y = -1", "Training data; y = +1", "Predictions; y = -1", "Predictions; y = +1",
            ], loc=2)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
