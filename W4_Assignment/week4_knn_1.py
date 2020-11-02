import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
import sklearn.preprocessing as prep
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("week4_set2.csv")

X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

# polynomial_features = prep.PolynomialFeatures(degree=5)
# X = polynomial_features.fit_transform(X)

k_range = np.linspace(1, 100, 20)
print("KRANGE: ", k_range)

mean_error = []
std_error = []
kf = KFold(n_splits=5)

for i, k in enumerate(k_range):
    knn_model = KNeighborsClassifier(n_neighbors=round(k))
    print("Trying k: ", round(k))
    temp = []

    for train, test in kf.split(X):
        knn_model.fit(X[train], y[train])
        predictions = knn_model.predict(X[test])
        temp.append(log_loss(y[test], predictions))
        print("--> ", log_loss(y[test], predictions))

    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

    fig = plt.figure(k)

    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(X1[test], X2[test], predictions, label="Training Data")
    # ax.plot_surface(X1_test, X2_test, predictions,
    #                 color="red", alpha=0.5, vmin=-2, vmax=2)

    ax.set_title(
        "3D Scatter plot of Predictions; k = " + str(round(k)))
    ax.set_xlabel("X1 Values")
    ax.set_ylabel("X2 Values")
    ax.set_zlabel("Output y")
    # ax.set_xlim([-2, 2])
    # ax.set_ylim([-2, 2])
    # ax.set_zlim([-5, 5])
    plt.show()


plt.figure(2)
plt.errorbar(k_range, mean_error, yerr=std_error, linewidth=3)
plt.xlabel("K - Number of neighbours")
plt.ylabel("Log loss")
plt.title("Log loss vs. 'K' Number of neighbours")
plt.show()

k_index_min_error = mean_error.index(min(mean_error))
print("MEAN ERROR: ", mean_error)
optimal_k = k_range[k_index_min_error]
print("OPTIMAL K: ", optimal_k)
