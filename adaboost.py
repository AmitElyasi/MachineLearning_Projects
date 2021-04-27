#################################
# Your name: Amit Elyasi
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """

    alpha_vals = []
    hypotheses = []
    n = len(X_train)
    D = [1 / n for i in range(n)]

    for t in range(T):
        h_t = WL(D, X_train, y_train)
        epsilon_t = weighted_error_t(X_train, y_train, D, h_t)
        w_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
        exponent = [(-1) * w_t * y_train[i] * prediction(X_train[i], h_t) for i in range(len(X_train))]
        next_D = np.multiply(D, np.exp(exponent))
        D = next_D / np.sum(next_D)
        alpha_vals.append(w_t)
        hypotheses.append(h_t)

    return hypotheses, alpha_vals


##############################################
# You can add more methods here, if needed.

def zo_loss(x, y, h):
    if prediction(x, h) == y:
        return 0
    return 1


def prediction(x, h):
    prediction = h[0]
    i = h[1]
    theta = h[2]
    if x[i] <= theta:
        return prediction
    return 0 - prediction


def sign(num):
    if num >= 0:
        return 1
    return -1


def empirical_error(data, labels, hypotheses, alpha_vals):
    errors = []
    n = len(data)
    sum = np.zeros(n)
    T = len(hypotheses)
    for t in range(T):
        error = 0
        for i in range(n):
            sum[i] += alpha_vals[t] * prediction(data[i], hypotheses[t])
            if sign(sum[i]) != labels[i]:
                error += 1
        errors.append(error / n)

    return errors


def loss(data, labels, hypotheses, alpha_vals):
    losses = []
    n = len(data)
    exponents = np.zeros(n)
    T = len(hypotheses)
    for t in range(T):
        for i in range(n):
            exponents[i] += (-1) * labels[i] * alpha_vals[t] * prediction(data[i], hypotheses[t])
        vec = np.exp(exponents)
        loss = sum(vec)
        losses.append(loss / n)

    return losses


def best_WL(D, X_train, y_train, sign_b):
    F_min = np.Infinity
    theta = 0
    min_index = 0
    d = len(X_train[0])
    m = len(X_train)

    for j in range(d):
        values = np.array([[X_train[i][j], y_train[i], D[i]] for i in range(m)])
        values = values[np.argsort(values[:, 0])]
        last = np.array([values[m - 1][0] + 1, 0, 0])
        values = np.vstack((values, last))
        F = 0
        for i in range(m):
            if values[i][1] == sign_b:
                F += values[i][2]
        if F < F_min:
            F_min = F
            theta = values[0][0] - 1
            min_index = j
        for i in range(m):
            F = F - sign_b * values[i][1] * values[i][2]
            if F < F_min and values[i][0] != values[i + 1][0]:
                F_min = F
                theta = 0.5 * (values[i][0] + values[i + 1][0])
                min_index = j

    return (min_index, theta, F_min)


def weighted_error_t(X_train, y_train, D, h):
    n = len(X_train)
    sum = 0
    for i in range(n):
        sum += D[i] * zo_loss(X_train[i], y_train[i], h)

    return sum


def WL(D, X_train, y_train):
    min_index_1, theta_1, F_min1 = best_WL(D, X_train, y_train, 1)
    min_index_2, theta_2, F_min2 = best_WL(D, X_train, y_train, -1)
    if F_min1 < F_min2:
        return 1, min_index_1, theta_1
    else:
        return -1, min_index_2, theta_2


##############################################
def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    h, alpha_vals = run_adaboost(X_train, y_train, 80)

    ##############################################
    # You can add more methods here, if needed.

    # a)
    train_error = empirical_error(X_train, y_train, h, alpha_vals)
    test_error = empirical_error(X_test, y_test, h, alpha_vals)
    X = np.arange(1, T + 1)
    plt.plot(X, train_error, 'b+', label='error of train set')
    plt.plot(X, test_error, 'ro', label='error of test set')
    plt.xlabel('t')
    plt.legend()
    # plt.show()

    # b)
    for i in range(10):
    # print(vocab[h[i][1]])

    # c)
    test_loss = loss(X_test, y_test, h, alpha_vals)
    train_loss = loss(X_train, y_train, h, alpha_vals)
    X = np.arange(1, T + 1)
    plt.plot(X, train_loss, 'b+', label='loss of train set')
    plt.plot(X, test_loss, 'ro', label='loss of test set')
    plt.xlabel('t')
    plt.legend()
    # plt.show()

    ##############################################


if __name__ == '__main__':
    main()
