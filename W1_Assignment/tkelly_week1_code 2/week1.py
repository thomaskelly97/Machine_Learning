import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# (a)(i) read the data in
df = pd.read_csv("week1.csv", comment='#')

X = np.array(df.iloc[:, 0])
y = np.array(df.iloc[:, 1])

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

x_mean = np.mean(X)
y_mean = np.mean(y)

x_std = np.std(X)
y_std = np.std(y)

# (a)(ii) Normalise the data
x_normalised = (X - x_mean) / x_std
y_normalised = (y - y_mean) / y_std

ones = np.ones([x_normalised.shape[0], 1])
x_normalised = np.concatenate([ones, x_normalised], 1)

theta = np.array([[1.0, 1.0]])

iterations = 1000
learning_rate = 0.1

log_cost_array = [0] * iterations
iteration_array = [0] * iterations


def costF(input_x, y, theta):
    calculate_inner = np.power(((input_x * theta) - y), 2)
    return np.sum(calculate_inner) / len(X)

# (a)(iii) Gradient Descent Algorithm Implementation


def gradientDescent(input_x, y, theta, alpha):
    for i in range(0, iterations):
        step_size = (-2*alpha/len(input_x)) * \
            np.sum(((input_x * theta) - y) * input_x, axis=0)
        theta = theta + step_size
        cost = costF(input_x, y, theta)
        log_cost_array[i] = cost  # Log the cost at this iteration
        iteration_array[i] = i  #  Log the value of this iteration
    return (theta, cost)


trained_theta_array, cost = gradientDescent(
    x_normalised, y_normalised, theta, learning_rate)

# Plot 1
# Cost vs Iterations
plt.figure(1)
plt.title("Cost Plot; alpha = 0.1")
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.scatter(iteration_array, log_cost_array, color='red')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()

# Plot 2
# Fitting linear plot to training data
plt.figure(2)
plt.title("Linear Regression Plot on Training Data")
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

re_normalised = (np.array(df.iloc[:, 0]).reshape(-1, 1) - x_mean) / x_std
plt.scatter(re_normalised, y_normalised, color='black')
axes = plt.gca()
xs = np.array(axes.get_xlim())
ys = trained_theta_array[0][0] + trained_theta_array[0][1] * xs
plt.plot(xs, ys, color='blue', linewidth=3)
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.legend(['linear regression plot', 'training data'])
plt.show()

# Plot 3
# Linear regression plot, with baseline model
plt.figure(3)
plt.title("Linear Regression Plot on Training Data")
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

plt.scatter(re_normalised, y_normalised, color='black')
axes = plt.gca()
plt.plot(xs, ys, color='blue', linewidth=3)
# Choose 0 because it is roughly around the mean of the y data
baseline_y = 0 + 0 * xs  #  Leaves us roughly in the middle
plt.plot(xs, baseline_y, color="red", linewidth=3)
plt.xlabel('X Input')
plt.ylabel('Y Output')
plt.legend(['linear regression plot', 'baseline model', 'training data'])
plt.show()
