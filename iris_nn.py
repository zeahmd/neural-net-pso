from sklearn.datasets import load_iris
from pso_numpy import *
import numpy as np


#load iris dataset..
data = load_iris()

#Store input & target in X and Y..
X = data.data
Y = data.target

#define no of nodes in each layer..
INPUT_NODES = 4
HIDDEN_NODES = 20
OUTPUT_NODES = 3

def one_hot_encode(Y):
    """
    create one-hot encoded vectors from target labels(Y).

    :param Y: int(N, )
    :return: int(N, C)
        Returns an array of shape(N, C) where C is number of classes.
    """
    num_unique = len(np.unique(np.array(Y)))
    zeros = np.zeros((len(Y), num_unique))
    zeros[range(len(Y)), Y] = 1
    return zeros

def softmax(logits):
    """
    Apply softmax function on logits and return probabilities.

    :param logits: double(N, C)
        Logits of each instance for each class.
    :return: double(N, C)
        probability for each class of each instance.
    """
    exps = np.exp(logits)
    return exps / np.sum(exps, axis=1, keepdims=True)

def Negative_Likelihood(probs, Y):
    """
    Calculates Negative Log Likelihood loss.

    :param probs: double(N, C)
        Probability of each instance for each class.
    :param Y: int(N, )
        Integer representation of each class.
    :return: double
        Returns value of loss calculated.
    """
    num_samples = len(probs)
    corect_logprobs = -np.log(probs[range(num_samples), Y])
    return np.sum(corect_logprobs) / num_samples

def Cross_Entropy(probs, Y):
    """
    Calculates Categorical Cross Entropy loss.

    :param probs: double(N, C)
        Probability of each instance for each class.
    :param Y: int(N, C)
        One-hot encoded representation of classes.
    :return: double
        Returns value of loss calculated.
    """
    num_samples = len(probs)
    ind_loss = np.max(-1 * Y * np.log(probs + 1e-12), axis=1)
    return np.sum(ind_loss) / num_samples

def forward_pass(X, Y, W):
    """
    Performs forward pass during Neural Net training.

    :param X: double(N, F)
        X is input where N is number of instances and F is number of features.
    :param Y: int(N, ) | int(N, C)
        Y is target where N is number of instances and C is number of classes in case of
        one-hot encoded target.
    :param W: double(N, )
        Weights where N is number of total weights(flatten).
    :return: double
        Returns loss of forward pass.
    """

    if isinstance(W, Particle):
        W = W.x

    w1 = W[0 : INPUT_NODES * HIDDEN_NODES].reshape((INPUT_NODES, HIDDEN_NODES))
    b1 = W[INPUT_NODES * HIDDEN_NODES:(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES].reshape((HIDDEN_NODES, ))
    w2 = W[(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES:(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES +\
        (HIDDEN_NODES * OUTPUT_NODES)].reshape((HIDDEN_NODES, OUTPUT_NODES))
    b2 = W[(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES): (INPUT_NODES *\
        HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES) + OUTPUT_NODES].reshape((OUTPUT_NODES, ))


    z1 = np.dot(X, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    logits = z2

    probs = softmax(logits)

    #Here we can calculate Categorical Cross Entropy from probs in case we have one-hot encoded vector
    #,or calculate Negative Log Likelihood from logits without one-hot encoded vector

    #We're going to calculate Negative Likelihood, because we didn't one-hot encoded Y target...
    return Negative_Likelihood(probs, Y)
    #return Cross_Entropy(probs, Y) #used in case of one-hot vector target Y...


def predict(X, W):
    """
    Performs forward pass during Neural Net test.

    :param X: double(N, F)
        X is input where N is number of instances and F is number of features.
    :param W: double(N, )
        Weights where N is number of total weights(flatten).
    :return: int(N, )
        Returns predicted classes.
    """

    w1 = W[0: INPUT_NODES * HIDDEN_NODES].reshape((INPUT_NODES, HIDDEN_NODES))
    b1 = W[INPUT_NODES * HIDDEN_NODES:(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES].reshape((HIDDEN_NODES,))
    w2 = W[(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES:(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES + \
        (HIDDEN_NODES * OUTPUT_NODES)].reshape((HIDDEN_NODES, OUTPUT_NODES))
    b2 = W[(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES): (INPUT_NODES * \
        HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES) + OUTPUT_NODES].reshape((OUTPUT_NODES,))

    z1 = np.dot(X, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    logits = z2

    probs = softmax(logits)
    Y_pred =  np.argmax(probs, axis=1)
    return Y_pred

def get_accuracy(Y, Y_pred):
    """
    Calcualtes accuracy.

    :param Y: int(N, )
        Correct labels.
    :param Y_pred: int(N, ) | double(N, C)
        Predicted labels of shape(N, ) or (N, C) in case of one-hot vector.
    :return: double
        Accuracy.
    """
    return (Y == Y_pred).mean()
    #return (np.argmax(Y, axis=1) == Y_pred).mean() #used in case of one-hot vector and loss is Negative Likelihood.

if __name__ == '__main__':
    no_solution = 100
    no_dim = (INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES) + OUTPUT_NODES
    w_range = (0.0, 1.0)
    lr_range = (0.0, 1.0)
    iw_range = (0.9, 0.9)  # iw -> inertial weight...
    c = (0.5, 0.3)  # c[0] -> cognitive factor, c[1] -> social factor...

    s = Swarm(no_solution, no_dim, w_range, lr_range, iw_range, c)
    #Y = one_hot_encode(Y) #Encode here...
    s.optimize(forward_pass, X, Y, 100, 1000)
    W = s.get_best_solution()
    Y_pred = predict(X, W)
    accuracy = get_accuracy(Y, Y_pred)
    print("Accuracy: %.3f"% accuracy)

