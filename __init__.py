#!/usr/bin/env python3

import os
import urllib.request
import gzip
import pickle as pkl
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))


def convert(image_in, label_in, out_csv, image_out_obj, label_out_obj, n):
    with gzip.open(image_in, "rb") as f, gzip.open(label_in, "rb") as l, open(
        out_csv, "w"
    ) as o, open(image_out_obj, "wb") as i_o, open(label_out_obj, "wb") as l_o:

        f.read(16)
        l.read(8)

        labels = np.zeros((n, 1), dtype=np.uint8)
        images = np.zeros((n, 28 * 28), dtype=np.uint8)

        for i in range(n):
            image = [ord(l.read(1))]
            image.extend(ord(f.read(1)) for j in range(28 * 28))
            o.write(",".join(str(pix) for pix in image) + "\n")

            labels[i, 0] = image[0]
            images[i, :] = image[1:]

        pkl.dump(images, i_o)
        pkl.dump(labels, l_o)


if not os.path.exists(f"{DIR}/data"):
    os.makedirs(f"{DIR}/data")

    if not os.path.isfile(f"{DIR}/data/mnist_train.csv"):
        print("MNIST train data not found. Downloading...")
        train_labels = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
        train_images = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"

        print("Retrieving training data...")
        urllib.request.urlretrieve(train_labels, f"{DIR}/data/tmp_train_labels.gz")
        urllib.request.urlretrieve(train_images, f"{DIR}/data/tmp_train_images.gz")
        print("Unzipping training data...")
        convert(
            f"{DIR}/data/tmp_train_images.gz",
            f"{DIR}/data/tmp_train_labels.gz",
            f"{DIR}/data/mnist_train.csv",
            f"{DIR}/data/mnist_train_images.pkl",
            f"{DIR}/data/mnist_train_labels.pkl",
            60000,
        )
        os.remove(f"{DIR}/data/tmp_train_labels.gz")
        os.remove(f"{DIR}/data/tmp_train_images.gz")
        print("Done.")

    if not os.path.isfile(f"{DIR}/data/mnist_test.csv"):
        print("MNIST test data not found. Downloading...")
        test_labels = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
        test_images = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"

        print("Retrieving testing data...")
        urllib.request.urlretrieve(test_labels, f"{DIR}/data/tmp_test_labels.gz")
        urllib.request.urlretrieve(test_images, f"{DIR}/data/tmp_test_images.gz")
        print("Unzipping testing data...")
        convert(
            f"{DIR}/data/tmp_test_images.gz",
            f"{DIR}/data/tmp_test_labels.gz",
            f"{DIR}/data/mnist_test.csv",
            f"{DIR}/data/mnist_test_images.pkl",
            f"{DIR}/data/mnist_test_labels.pkl",
            10000,
        )
        os.remove(f"{DIR}/data/tmp_test_labels.gz")
        os.remove(f"{DIR}/data/tmp_test_images.gz")
        print("Done.")
