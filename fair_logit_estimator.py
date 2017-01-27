""" A skikit-learn compatible, provably fair binary classifier using logistic regression """

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
from sklearn.utils.fixes import expit
import fair_utils
# from sklearn.utils.multiclass import unique_labels

# pylint: disable=invalid-name
# pylint: disable=no-self-use
# pylint: disable=attribute-defined-outside-init
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments

def _intercept_dot(w, X, y):
    """Computes y * np.dot(X, w).
    It takes into consideration if the intercept should be fit or not.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    Returns
    -------
    w : ndarray, shape (n_features,)
        Coefficient vector without the intercept weight (w[-1]) if the
        intercept should be fit. Unchanged otherwise.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Unchanged.
    yz : float
        y * np.dot(X, w).
    """
    c = 0.
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]

    z = safe_sparse_dot(X, w) + c
    yz = y * z
    return w, c, yz

def _logistic_loss_and_grad(w, X, y, alpha, sample_weight=None):
    """Computes the logistic loss and gradient.

    Copied from scikit-learn/scikit-learn
    commit 7f224f8
    file: sklearn/linear_model/logistic.py
    url: https://github.com/scikit-learn/scikit-learn/
         blob/14031f6/sklearn/linear_model/logistic.py#L78



    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    Returns
    -------
    out : float
        Logistic loss.
    grad : ndarray, shape (n_features,) or (n_features + 1,)
        Logistic gradient.
    """
    n_samples, n_features = X.shape
    grad = np.empty_like(w)

    w, _c, yz = _intercept_dot(w, X, y)

    if sample_weight is None:
        sample_weight = np.ones(n_samples)

    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(sample_weight * log_logistic(yz)) + .5 * alpha * np.dot(w, w)

    z = expit(yz)
    z0 = sample_weight * (z - 1) * y

    grad[:n_features] = safe_sparse_dot(X.T, z0) + alpha * w

    # Case where we fit the intercept.
    if grad.shape[0] > n_features:
        grad[-1] = z0.sum()
    return out, grad

class FairLogitEstimator(BaseEstimator, ClassifierMixin):
    """ A logistic regression estimator that also takes into account fairness
        over a (binary) sensitive attribute
    Parameters
    ----------
    constraint : str, optional
        Specifies which constraint to use: "fairness" minimizes the loss function
        while constraining covariance between the sensitive attribute and distance
        from the decision boundary. "accuracy" minimizes covariance between
        the sensitive attribute and distance from the decision boundary while
        constraining the loss function.
    sensitive_col_idx : int, optional
        Specifies which column of X (during fitting) contains the sensitive
        attribute.
    covariance_tolerance : float, optional
        Threshhold below which the covariance should be constrained. Only
        applicable if constraint="fairness".
    covariance_tolerance : float, optional
        If constraint="accuracy", the loss function is constrained to
        (1+covariance_tolerance) the loss function of the optimal parameters
        without regard to fairness.
    """
    def __init__(self, constraint='fairness', sensitive_col_idx=0,
                 covariance_tolerance=0, accuracy_tolerance=0):
        self.constraint = constraint
        self.sensitive_col_idx = sensitive_col_idx
        self.covariance_tolerance = covariance_tolerance
        self.accuracy_tolerance = accuracy_tolerance


    def fit(self, X, y):
        """A reference implementation of a fitting function
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. By convention, we'll assume that
            the protected attr is last in each row.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        # check if X & y have the correct shape
        X, y = check_X_y(X, y, y_numeric=True)
        # Store the classes seen during fit
        # self.classes_ = unique_labels(y)
        if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise ValueError("Training data has non-numeric dtype. X: %s y: %s"
                             % (X.dtype, y.dtype))

        if len(np.unique(y)) > 2:
            raise ValueError("Only two y values are permissible for a binary logit classifier.")

        # remove sensitive column to separate array
        x_control = {'foo': X[:, self.sensitive_col_idx]}
        X = np.delete(X, self.sensitive_col_idx, 1)

        apply_fairness_constraints = 1 if self.constraint == 'fairness' else 0
        apply_accuracy_constraint = 1 if self.constraint == 'accuracy' else 0
        sep_constraint = 0 # currently not using this mode
        sensitive_attrs = ["foo"] # doesn't matter what we name this since there
                                  # can only be one
        sensitive_attrs_to_cov_thresh = {'foo': self.covariance_tolerance}

        self.w_ = fair_utils.train_model(X, y, x_control, _logistic_loss_and_grad,
                                         apply_fairness_constraints,
                                         apply_accuracy_constraint,
                                         sep_constraint,
                                         sensitive_attrs, sensitive_attrs_to_cov_thresh,
                                         self.accuracy_tolerance)

        # Return the estimator
        return self

    def predict(self, X):
        """ Predicts labels for all samples in X

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        check_is_fitted(self, ['w_'])
        X = check_array(X)

        # remove sensitive col from test input
        X = np.delete(X, self.sensitive_col_idx, 1)

        return np.sign(np.dot(X, self.w_))

    def predict_proba(self, X):
        """ Gives probability estimates of each label, for all samples in X

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self, ['w_'])
        X = check_array(X)

        # remove sensitive col from test input
        X = np.delete(X, self.sensitive_col_idx, 1)

        probs = np.dot(X, self.w_)

        return np.array([(probs*-1+1)/2, (probs+1)/2])

    def boundary_distances(self, X):
        """ Returns the dot product of each sample in X
        with w.
        """
        check_is_fitted(self, ['w_'])
        X = check_array(X)

        # remove sensitive col from test input
        X = np.delete(X, self.sensitive_col_idx, 1)

        return np.dot(X, self.w_)
