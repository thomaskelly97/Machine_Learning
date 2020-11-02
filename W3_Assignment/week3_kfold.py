import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import errorbar
import math


dataframe = pd.read_csv("week3_data.csv")

X1 = dataframe.iloc[:, 0]
X2 = dataframe.iloc[:, 1]
X = np.column_stack((X1, X2))

y = dataframe.iloc[:, 2]

polynomial_features = prep.PolynomialFeatures(degree=5)
new_features = polynomial_features.fit_transform(X)

C = 1
split_values = [2, 5, 10, 25, 50, 100]
mean_array = []*len(split_values)
stan_dev_array = []*len(split_values)
for i, split in enumerate(split_values):
    alpha = 1/(2*C)  #  As specified in assignment notes, sklearn alpha = 1/2C
    # scores = cross_val_score(lassoModel, X, y, cv=5,
    #                          scoring="neg_mean_squared_error")
    print("--- SPLIT = ", split, " ---")
    kf = KFold(n_splits=split)
    loopCount = 0
    sum_errors = 0
    error_array = []*split
    for train, test in kf.split(new_features):

        loopCount = loopCount + 1
        lassoModel = linear_model.Lasso(alpha=alpha, random_state=0)
        lassoModel.fit(new_features[train], y[train])

        predictions = lassoModel.predict(new_features[test])
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
errorbar(split_values, mean_array, yerr=stan_dev_array)
plt.xlabel("K-Fold 'K' Value")
plt.ylabel("Mean Square Error")
plt.title("Mean Squared Error vs. K-Fold Value; C = 1")
plt.show()
