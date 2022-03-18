#!/usr/bin/env python3

import pickle as pkl
import os

DIR = os.path.dirname(os.path.realpath(__file__))


def load_train_numpy():
    with open(f"{DIR}/data/mnist_train_images.pkl", "rb") as train_i, open(
        f"{DIR}/data/mnist_train_labels.pkl", "rb"
    ) as train_l:
        return pkl.load(train_i), pkl.load(train_l)


def load_test_numpy():
    with open(f"{DIR}/data/mnist_test_images.pkl", "rb") as test_i, open(
        f"{DIR}/data/mnist_test_labels.pkl", "rb"
    ) as test_l:
        return pkl.load(test_i), pkl.load(test_l)


def csv_path():
    return f"{DIR}/data/mnist_train.csv", f"{DIR}/data/mnist_test.csv"
