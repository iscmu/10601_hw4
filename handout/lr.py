import numpy as np
import argparse
# from feature import load_tsv_dataset

def load_tsv_dataset(file):
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='float')
    return dataset

def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> None:
    # TODO: Implement `train` using vectorization
    for e in range(num_epoch):
        # theta -= learning_rate * (-(y - sigmoid( X @ theta)) @ X)
        for i in range(len(X)):
            arg = X[i] @ theta
            theta -= learning_rate * ((-(y[i] - sigmoid( arg )) * X[i]))
    

def predict(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray      # shape (N, D) where N is num of examples
) -> np.ndarray:
    # TODO: Implement `predict` using vectorization
    # pass
    return sigmoid(X @ theta)


def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    # TODO: Implement `compute_error` using vectorization
    return 1 - ((y_pred >= 0.5) == y).sum() / len(y_pred)


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    data_train = load_tsv_dataset(args.train_input)
    data_test = load_tsv_dataset(args.test_input)
    data_val = load_tsv_dataset(args.validation_input)

    # Training
    X = data_train[:,1:]
    y = data_train[:, 0]

    # Folding the intercept vector
    theta = np.zeros(X.shape[1] + 1, dtype=float)
    X = np.pad(X, ((0, 0), (0, 1)), mode='constant', constant_values=1)

    train(theta, X, y, args.num_epoch, args.learning_rate)
    y_pred = predict(theta, X)
    tr_err = compute_error(y_pred, y)

    # Validation
    Xval = data_val[:, 1:]
    yval = data_val[:, 0]
    Xval = np.pad(Xval, ((0, 0), (0, 1)), mode='constant', constant_values=1)
    y_pred_val = predict(theta, Xval)
    val_err = compute_error(y_pred_val, yval)

    # Test
    Xtest = data_test[:,1:]
    ytest = data_test[:, 0]
    Xtest = np.pad(Xtest, ((0, 0), (0, 1)), mode='constant', constant_values=1)
    y_pred_test = predict(theta, Xtest)
    test_err = compute_error(y_pred_test, ytest)

    def write_result(f, labels):
        for i in labels:
            f.write(str(int(i)) + '\n')

    with open(args.train_out, 'w') as f:
        write_result(f, y_pred)
    with open(args.test_out, 'w') as f:
        write_result(f, y_pred_test)
    with open(args.metrics_out, 'w') as f:
        f.write("error(train): " + str(f"{tr_err:.6f}" + "\n"))
        f.write("error(test): " + str(f"{test_err:.6f}" + "\n"))


    # VERIFICATION 
    




