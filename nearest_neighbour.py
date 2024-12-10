from typing import Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

from Classifier import Classifier


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    input_to_labels = {tuple(x_train[i]): y_train[i] for i in range(x_train.shape[0])}

    classifier = Classifier(k, input_to_labels)

    return classifier


def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    y_testprediction = np.array([])
    for xi in x_test:
        labels: dict[int: int] = {i: 0 for i in range(10)}
        distances: dict[Tuple: float] = {
            key: np.linalg.norm(xi - np.array(key))
            for key in classifier.input_to_labels.keys()
        }

        knn = dict(sorted(distances.items(), key=lambda x: x[1])[:classifier.k])
        for x_train in knn:
            labels[classifier.input_to_labels[x_train]] += 1
        max_label = max(labels, key=labels.get)
        y_testprediction = np.append(y_testprediction, [max_label])

    return np.array(y_testprediction, dtype=int).reshape(-1, 1)


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


def tests_question2():
    data = np.load('mnist_all.npz')

    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    # -----------------------------------------------Question_2Part_a---------------------------------------------------

    # avg_list: List[float] = []
    # min_error_list = []
    # max_error_list = []
    # for i in range(10, 101, 10):
    #     errors: List[float] = []
    #     for j in range(1, 11):
    #         x_train, y_train = gensmallm([train2, train3, train5, train6], [2, 3, 5, 6], i)
    #         x_test, y_test = gensmallm([test2, test3, test5, test6], [2, 3, 5, 6], 50)
    #         classifier = learnknn(1, x_train, y_train)
    #         y_testprediction = predictknn(classifier, x_test)
    #         error = np.mean(y_test.flatten() != y_testprediction.flatten())
    #         errors.append(error)
    #         print(f"Sample size: {i} iteration: {j} error: {error}")
    #     avg: float = np.mean(errors)
    #     avg_list.append(round(avg, 2))
    #     min_error_list.append(min(errors))
    #     max_error_list.append(max(errors))
    # print(avg_list)
    #
    # sample_sizes = list(range(10, 101, 10))
    # lower_error = np.array(avg_list) - np.array(min_error_list)
    # upper_error = np.array(max_error_list) - np.array(avg_list)
    #
    # ax = plt.axes()
    # ax.errorbar(sample_sizes, avg_list, yerr=[lower_error, upper_error], marker='o', label='Average Error')
    # ax.set(xlim=(0, 120), ylim=(0, 1), xlabel='sample_size', ylabel='average_error')
    # ax.legend()
    # plt.grid(True)
    # plt.show()

    # -----------------------------------------------Question_2Part_d---------------------------------------------------
    avg_list: List[float] = []
    min_error_list = []
    max_error_list = []
    errors: List[float] = []
    for i in range(1, 12):
        for j in range(1, 11):
            x_train, y_train = gensmallm([train2, train3, train5, train6], [2, 3, 5, 6], 200)
            x_test, y_test = gensmallm([test2, test3, test5, test6], [2, 3, 5, 6], 50)
            classifier = learnknn(i, x_train, y_train)
            y_testprediction = predictknn(classifier, x_test)
            error = np.mean(y_test.flatten() != y_testprediction.flatten())
            errors.append(error)
            print(f"k = {i} Sample size: 200 error: {error}")
        avg: float = np.mean(errors)
        avg_list.append(round(avg, 2))
        min_error_list.append(min(errors))
        max_error_list.append(max(errors))
    print(avg_list)

    k_sizes = list(range(1, 12, 1))
    lower_error = np.array(avg_list) - np.array(min_error_list)
    upper_error = np.array(max_error_list) - np.array(avg_list)

    ax = plt.axes()
    ax.errorbar(k_sizes, avg_list, yerr=[lower_error, upper_error], marker='o', label='Average Error')
    ax.set(xlim=(0, 12), ylim=(0, 1), xlabel='k_size', ylabel='average_error')
    ax.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors

    # k = 1
    # x_train = np.array([[1, 2], [3, 4], [5, 6]])
    # y_train = np.array([1, 0, 1])
    # classifier = learnknn(k, x_train, y_train)
    #
    # x_test = np.array([[10, 11], [3.1, 4.2], [2.9, 4.2], [5, 6]])
    # y_testprediction = predictknn(classifier, x_test)
    # print(y_testprediction)

    # simple_test()

    tests_question2()
