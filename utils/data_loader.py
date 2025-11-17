from mnist import MNIST
import numpy as np


def load_data(path: str = "./mnist") -> tuple[np.ndarray]:
    """
    loads data and converts images to np arrays normalized between 0 and 1, and labels to one hot encodig, 
    meaning all labels are an ndarray of tuples with len 10, where a 1 on an index represent what number it is.
    number 5 will have the label [0,0,0,0,0,1,0,0,0,0]

    returns X_train, y_train_oh, X_test, y_test_oh
    """

    mndata = MNIST(path)
    mndata.gz = True
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    X_train = np.array(train_images) / 255
    X_test = np.array(test_images) / 255
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    y_train_oh = np.eye(10)[y_train]
    y_test_oh = np.eye(10)[y_test]

    return X_train, y_train_oh, X_test, y_test_oh