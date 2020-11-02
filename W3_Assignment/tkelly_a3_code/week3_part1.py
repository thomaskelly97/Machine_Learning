import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


dataframe = pd.read_csv("week3_data.csv")

X1 = dataframe.iloc[:, 0]
X2 = dataframe.iloc[:, 1]
X = np.column_stack((X1, X2))

y = dataframe.iloc[:, 2]

polynomial_features = prep.PolynomialFeatures(degree=5)
new_features = polynomial_features.fit_transform(X)

# ---- (i)(c) Setup Xtest data
Xtest = []
grid = np.linspace(-2, 2)
for i in grid:
    for j in grid:
        Xtest.append([i, j])
Xtest = np.array(Xtest)
Xtest = polynomial_features.fit_transform(Xtest)
Xtest_columns = np.column_stack(Xtest)
X1_test = Xtest_columns[1]
X2_test = Xtest_columns[2]

c_values = [1, 2, 5, 10, 100, 1000, 10000]
thetas = []
for i, C in enumerate(c_values):
    alpha = 1/(2*C)  #  As specified in assignment notes, sklearn alpha = 1/2C
    lassoModel = linear_model.Lasso(alpha=alpha, random_state=0)

    lassoModel.fit(new_features, y)
    thetas.append(lassoModel.coef_)
    print("XTEST: ", Xtest)
    predictions = lassoModel.predict(Xtest)
    predictions = np.reshape(predictions, (50, 50))
    X1_test = np.reshape(X1_test, (50, 50))
    X2_test = np.reshape(X2_test, (50, 50))
    print("PREDICTIONS: ", predictions)
    print("--> COEF: ", lassoModel.coef_)
    print("-->INTERCEPT: ", lassoModel.intercept_)

    # (i) (a)
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(X1, X2, y, label="Training Data")
    ax.plot_surface(X1_test, X2_test, predictions,
                    color="red", alpha=0.5, vmin=-2, vmax=2)

    ax.set_title(
        "3D Scatter plot of Training Data with Regression Model Surface; C = " + str(C))
    ax.set_xlabel("X1 Values")
    ax.set_ylabel("X2 Values")
    ax.set_zlabel("Output y")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-5, 5])
    fake2Dline = mpl.lines.Line2D(
        [0], [0], linestyle="none", c='r', marker='o')
    ax.legend([fake2Dline, sc], ['Regression surface',
                                 'Training Data'], numpoints=1)
    plt.show()

thetas = np.array(thetas)
theta_for_each_c = np.column_stack(thetas)


# (i)(b)
count = 1
for k in enumerate(theta_for_each_c):
    plt.figure(2)
    plt.plot(c_values, k[1])
    plt.title("Comparing model parameter theta-" +
              str(count) + " under each value of C")
    plt.ylabel("Theta " + str(count))
    plt.xlabel("C Value")
    count = count + 1
    plt.show()
