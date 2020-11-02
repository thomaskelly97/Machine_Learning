from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("week1.csv", comment='#')

X = np.array(df.iloc[:, 0])
y = np.array(df.iloc[:, 1])

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

x_mean = np.mean(X)
y_mean = np.mean(y)

x_std = np.std(X)
y_std = np.std(y)

x_normalised = (X - x_mean) / x_std
y_normalised = (y - y_mean) / y_std

X_train, X_test, y_train, y_test = train_test_split(
    x_normalised, y_normalised, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Plot 1
plt.figure(2)
plt.title("Linear Regression Plot With Sklearn")
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

plt.scatter(x_normalised, y_normalised, color='black')
axes = plt.gca()
xs = np.array(axes.get_xlim())
ys = regressor.intercept_ + regressor.coef_ * xs
plt.plot(xs, ys[0], color='blue', linewidth=3)
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.legend(['linear regression plot', 'training data'])
plt.show()
