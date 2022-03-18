# MNIST loader

A simple library to load data from the mnist database in the form of numpy arrays or csv files.

Downloading and decompressing the files may take some time on first usage.

## Usage

```python
from mnist_loader import loader

x_train, y_train = loader.load_train_numpy()
x_test, y_test = loader.load_train_numpy()
```
