import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def plot_func(num_of_attr, interval_vector, w):
    gamma_length = [len(i) for i in interval_vector]
    for j in range(num_of_attr):  # 第j个准则
        criterion = w[j * (gamma_length[j] - 1): (j * (gamma_length[j] - 1) + (gamma_length[j] - 1))]
        criterion_vector = []
        criterion_vector.append(0.0)
        criterion_vector.append(criterion[0])
        # print('*****************')
        # print(criterion)
        for index in range(2, len(criterion) + 1):
            criterion_vector.append(sum(criterion[0:index]))
        plt.subplot(4, ceil(num_of_attr / 4), j + 1)
        plt.plot(interval_vector[j], criterion_vector)
    plt.show()

def plot_poly(coefficients, degree, num_of_attr):
    X = np.linspace(0.0, 1.0, 21)
    Y = []

    Y_each_model = []
    for item in range(0, num_of_attr):
        b = [coefficients[item * degree: item * degree + degree]]  # K is the polynomial degree
        b = list(reversed(b[0]))
        b.append(0.0)
        b = np.ravel(np.array(b).astype(float))
        # print(b)
        poly = np.poly1d(b, False)
        temp = poly(X)
        # print(temp)
        Y_each_model.append(temp)
    Y_each_model = np.array(Y_each_model)
    for j in range(num_of_attr):
        plt.subplot(4, ceil(num_of_attr / 4), j + 1)
        plt.plot(X, Y_each_model[j])
    plt.show()

def plot_ploy_func(num_of_attr, interval_vector, w, coefficients, degree,):
    gamma_length = [len(i) for i in interval_vector]

    Y_each_model = []
    X_each_model = []
    # For orginal ploynomial
    for item in range(0, num_of_attr):
        b = [coefficients[item * degree: item * degree + degree]]  # K is the polynomial degree
        b = list(reversed(b[0]))
        b.append(0.0)
        b = np.ravel(np.array(b).astype(float))
        # print(b)
        poly = np.poly1d(b, False)
        X = np.linspace(0, np.max(interval_vector[item]), 50, endpoint=True)
        X_each_model.append(X)
        temp = poly(X)
        # print(temp)
        Y_each_model.append(temp)
    Y_each_model = np.array(Y_each_model)
    X_each_model = np.array(X_each_model)
    # For obtained functions
    for j in range(num_of_attr):  # 第j个准则
        criterion = w[j * (gamma_length[j] - 1): (j * (gamma_length[j] - 1) + (gamma_length[j] - 1))]
        criterion_vector = []
        criterion_vector.append(0.0)
        criterion_vector.append(criterion[0])
        # print('*****************')
        # print(criterion)
        for index in range(2, len(criterion) + 1):
            criterion_vector.append(sum(criterion[0:index]))
        plt.subplot(4, ceil(num_of_attr / 4), j + 1)
        plt.plot(X_each_model[j], Y_each_model[j])
        plt.plot(interval_vector[j], criterion_vector, 'k--')
    plt.show()