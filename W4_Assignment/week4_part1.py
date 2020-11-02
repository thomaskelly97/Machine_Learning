import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as prep
from sklearn.model_selection import KFold

df = pd.read_csv("week4_set1.csv")

X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]


dataframe = pd.DataFrame({'column1': X1, 'column2': X2, 'column3': y})
minusDF = []
postivieDF = []

for grouped_y, grouping_x in dataframe.groupby(["column3"]):
    if grouped_y == -1:
        minusDF = grouping_x  # X rows that have y = -1
    else:
        positiveDF = grouping_x  #  X rows that have y = +1

polynomial_features = prep.PolynomialFeatures(degree=3)  #  use initial q
new_features = polynomial_features.fit_transform(X)

initial_C = 1
split_values = [2, 5, 10, 25, 50, 100]
mean_array = []*len(split_values)
stan_dev_array = []*len(split_values)
for i, split in enumerate(split_values):

    print("--- SPLIT = ", split, " ---")
    kf = KFold(n_splits=split)
    loopCount = 0
    sum_errors = 0
    error_array = []*split
    for train, test in kf.split(new_features):

        loopCount = loopCount + 1
        logRegressionModel = LogisticRegression(
            C=initial_C, penalty='l1', solver="saga", max_iter=7000, random_state=0)
        logRegressionModel.fit(new_features[train], y[train])

        predictions = logRegressionModel.predict(new_features[test])
        # print("Intercept: ",
        #       lassoModel.intercept_, "\nCOEF: ", lassoModel.coef_, "ERROR: ", mean_squared_error(y[test], predictions))
        mse = mean_squared_error(y[test], predictions)

        sum_errors = sum_errors + mse
        error_array.append(mse)
    mean = sum_errors / split
    variance = np.var(error_array)
    print("Mean: ", mean, "- Variance: ", variance)
    mean_array.append(mean)
    stan_dev_array.append(np.array(error_array).std())

plt.figure(1)
plt.errorbar(split_values, mean_array, yerr=stan_dev_array)
plt.xlabel("K-Fold 'K' Value")
plt.ylabel("Mean Square Error")
plt.title("Mean Squared Error vs. K-Fold Value; C = 1")
plt.show()

index_of_min_k = mean_array.index(min(mean_array))
print("MIN INDEX: ", index_of_min_k)
optimal_k = split_values[index_of_min_k]
print("optimal K", optimal_k)


def determine_optimal_q(c):
    kf = KFold(n_splits=optimal_k)
    mean_error = []
    std_error = []
    q_range = [1, 2, 3, 4, 5, 6]
    for i, q in enumerate(q_range):
        print("--- Trying Q Value: ", q)
        polynomial_features = prep.PolynomialFeatures(degree=q)
        new_features = polynomial_features.fit_transform(X)
        logRegressionModel = LogisticRegression(
            C=1, penalty='l2', solver="saga", max_iter=7000, random_state=0)
        temp = []

        for train, test in kf.split(new_features):
            logRegressionModel.fit(new_features[train], y[train])
            predictions = logRegressionModel.predict(new_features[test])
            temp.append(mean_squared_error(y[test], predictions))

        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.figure(2)
    plt.errorbar(q_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("Polynomial Degree 'q'")
    plt.ylabel("Mean square error")
    plt.title("Mean Square Error vs. Polynomial Degree Q")
    # plt.show()
    curr_min = min(mean_error)
    indexOfMinimum = [i for i, j in enumerate(mean_error) if j == curr_min]
    return indexOfMinimum[0] + 1  #  take first element for simplest model


c_values = [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000]


def determine_optimal_C():
    kf = KFold(n_splits=optimal_k)
    mean_error = []
    std_error = []
    for i, C in enumerate(c_values):
        print("--- Trying C Value: ", C)
        polynomial_features = prep.PolynomialFeatures(degree=5)
        new_features = polynomial_features.fit_transform(X)
        logRegressionModel = LogisticRegression(
            C=C, penalty='l1', solver="saga", max_iter=7000, random_state=0)

        temp = []
        for train, test in kf.split(new_features):
            logRegressionModel.fit(new_features[train], y[train])
            predictions = logRegressionModel.predict(new_features[test])
            temp.append(mean_squared_error(y[test], predictions))

        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.figure(3)
    plt.errorbar(c_values, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("C Value")
    plt.ylabel("Mean square error")
    plt.title("Mean Square Error vs. C Values")
    plt.xlim([0, 20])
    # plt.show()
    curr_min = min(mean_error)
    indexOfMinimum = [i for i, j in enumerate(mean_error) if j == curr_min]
    print("OPTIMAL C must be: ", c_values[indexOfMinimum[0]])
    #  take first element for simplest model
    return indexOfMinimum[0]


def run(q, c):
    polynomial_features = prep.PolynomialFeatures(degree=q)
    new_features = polynomial_features.fit_transform(X)
    logRegressionModel = LogisticRegression(
        C=c_values[c], penalty='l1', solver="saga", max_iter=7000, random_state=0)
    logRegressionModel.fit(new_features, y)
    predictions = logRegressionModel.predict(new_features)

    prediction_dataframe = pd.DataFrame(
        {'column1': X1, 'column2': X2, 'column3': predictions})
    minus_pred_DF = []
    positive_pred_DF = []

    for grouped_y, grouping_x in prediction_dataframe.groupby(["column3"]):
        if grouped_y == -1:
            minus_pred_DF = grouping_x  # X rows that have y = -1
        else:
            positive_pred_DF = grouping_x  #  X rows that have y = +1

    # if minus_pred_DF == []:  #  If minus_pred is empty, make sure its an empty dataframe
    #     minus_pred_DF = pd.DataFrame(
    #         {'column1': [], 'column2': [], 'column3': []})

    # PLOT
    plt.figure(1)
    plt.title(
        "Visualisation of Training Data with Predicted Target Values")
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

    # plt.xlim(-1, 1)
    plt.legend(["Training data; y = -1", "Training data; y = +1",
                "Predictions; y = -1", "Predictions; y = +1"])

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


index_of_optimal_c = determine_optimal_C()
optimal_q = determine_optimal_q(index_of_optimal_c)

while(True):
    print("Using Q: ", optimal_q, "\nUsing C: ", c_values[index_of_optimal_c])
    run(optimal_q, index_of_optimal_c)
    increase_q = input(
        "Based on visual analysis, do you want to increase q? y/n")
    if (increase_q == "y"):
        optimal_q = optimal_q + 1
    else:
        break
