import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode
from sklearn.metrics import cluster, normalized_mutual_info_score, adjusted_rand_score


def accuracy_score(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)

    # Find optimal one-to-one mapping between cluster labels and true labels
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


def purity_score(y_true, y_pred):
    """
    Calculate clustering purity.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        purity, in [0,1]
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)

    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def nmi_score(y_true, y_pred):
    """
    Calculate clustering normalized mutual information (NMI).
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        NMI, in [0,1]
    """

    # return NMI
    return normalized_mutual_info_score(y_true, y_pred)


def cluster_metric(y_true, y_pred):
    """
    Calculate clustering metrics, including accuracy, NMI and purity.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        [accuracy, NMI, purity], in [0,1]
    """
    # compute accuracy, NMI and purity
    acc = accuracy_score(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred)
    pur = purity_score(y_true, y_pred)

    # return metrics
    return acc, nmi, pur
