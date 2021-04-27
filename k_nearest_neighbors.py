import numpy.random
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt

#a
def knn(images, labels, query, k):
    distance = []
    for i in range(len(images)):
        distance.append((numpy.linalg.norm(images[i] - query), labels[i]))

    k_nearest = sorted(distance, key=lambda item: item[0])[:k]
    k_labels = [item[1] for item in k_nearest]

    return most_frequent(k_labels)


def most_frequent(items):
    if len(items) == 0:
        return None
    i = 0
    frequencies = {}
    while i < len(items):
        if items[i] not in frequencies:
            frequencies[items[i]] = 1
        else:
            frequencies[items[i]] += 1
        i += 1
    return max(frequencies, key=frequencies.get)


def test_accuracy_percentage_knn(n, k, training_images, training_labels, test_images, test_labels):
    images = training_images[:n]
    labels = training_labels[:n]
    successes = 0

    for i in range(len(test_images)):
        prediction = knn(images, labels, test_images[i], k)
        if prediction == test_labels[i]:
            successes += 1

    return (successes / len(test_images)) * 100


def prediction_plot(training_images, training_labels, test_images, test_labels):
    y = []
    max_accuracy = 0
    for k in range(1,101):
        accuracy = test_accuracy_percentage_knn(1000, k, training_images, training_labels, test_images, test_labels)
        y.append(accuracy)
        if accuracy > y[max_accuracy]:
            max_accuracy = k

    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.plot(range(1,101), y)
    plt.show()

    y = []
    for n in range(100,5001,100):
        y.append(test_accuracy_percentage_knn(n, 1, training_images, training_labels, test_images, test_labels))

    plt.xlabel("n")
    plt.ylabel("accuracy")
    plt.plot(range(100,5001,100), y)
    plt.show()


mnist = fetch_openml('mnist_784')
data = mnist['data']
labels = mnist['target']

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

# b
print("b) " + str(test_accuracy_percentage_knn(1000, 10, train, train_labels, test, test_labels)) + "%")

#c+d
prediction_plot(train, train_labels, test, test_labels)
