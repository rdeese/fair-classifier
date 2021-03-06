""" A skikit-learn compatible, provably fair binary classifier using logistic regression """

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
from sklearn.utils.fixes import expit
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize # for loss func minimization

# from sklearn.utils.multiclass import unique_labels

# pylint: disable=invalid-name
# pylint: disable=no-self-use
# pylint: disable=attribute-defined-outside-init
# pylint: disable=too-many-arguments

def _intercept_dot(w, X, y):
    """
    Copied from scikit-learn/scikit-learn
    commit 7f224f8
    file: TODO fill me in
    url: TODO fill me in

    Computes y * np.dot(X, w).
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

def _separate_sensitive_attrs(X, sensitive_col_idx):
    """
    Parameters
    ----------
    X : array-like or sparse matrix of shape = [n_samples, n_features]
        The training input samples.
    sensitive_col_idx : array-like, shape = [n_sensitive attrs]
        Specifies which column(s) of X contain(s) the sensitive
        attribute.
    Returns
    -------
    unsensitive_x : array of shape = [n_samples, n_features-n_senstitive_attrs]
    sensitive_x : array of shape = [n_samples, n_senstitive_attrs]
    """
    sensitive_x = X[:, sensitive_col_idx]
    unsensitive_x = np.delete(X, sensitive_col_idx, 1)

    return unsensitive_x, sensitive_x

def _get_fairness_constraint(unsensitive_x, sensitive_attr_vals,
                             correlation_tolerance):
    def constraint_fn(w, unsensitive_x, sensitive_attr_vals,
                      correlation_tolerance):
        """ Function passed to the minimizer, implements Eq. 2 from the paper """
        y = np.dot(w, unsensitive_x.T)
        debiased_sensitive_attr_vals = sensitive_attr_vals - np.mean(sensitive_attr_vals)
        covariance = np.dot(debiased_sensitive_attr_vals,
                            y) / float(len(sensitive_attr_vals))
        # We actually want to use the correlation, since it's normalized
        correlation = covariance / (np.std(debiased_sensitive_attr_vals) * np.std(y))

        # non-negative (constraint satisfied) if positive

        return correlation_tolerance - abs(correlation)

    return {
        'type': 'ineq',
        'fun': constraint_fn,
        'args': (unsensitive_x, sensitive_attr_vals, correlation_tolerance)
    }

def _get_fairness_constraints(unsensitive_x, sensitive_x, correlation_tolerance):
    enc = OneHotEncoder(sparse=False) # output transformed data as an array
    enc.fit(sensitive_x)
    encoded_x = enc.transform(sensitive_x)

    # map correlation tolerances to encoded columns
    # nested, unmasked
    encoded_correlation_tolerance = [enc.n_values_[ind]*[val]
                                     for ind, val in enumerate(correlation_tolerance)]
    # flattened, unmasked
    encoded_correlation_tolerance = [item for sublist in encoded_correlation_tolerance
                                     for item in sublist]
    encoded_correlation_tolerance = np.take(encoded_correlation_tolerance,
                                            enc.active_features_)

    return [_get_fairness_constraint(unsensitive_x,
                                     encoded_x[:, attr_index],
                                     encoded_correlation_tolerance[attr_index])
            for attr_index in range(encoded_x.shape[1])]




def _train_model_for_fairness(X, y, sensitive_col_idx,
                              correlation_tolerance):
    unsensitive_x, sensitive_x = _separate_sensitive_attrs(X, sensitive_col_idx)
    constraints = _get_fairness_constraints(unsensitive_x, sensitive_x,
                                            correlation_tolerance)

    alpha = 1.0 # alpha = 1 / C, a regularization parameter
    w = minimize(fun=_logistic_loss_and_grad,
                 x0=np.random.rand(unsensitive_x.shape[1],),
                 args=(unsensitive_x, y, alpha),
                 method='SLSQP',
                 options={"maxiter": 10000},
                 constraints=constraints,
                 jac=True
                )

    return w.x

def _train_model_without_fairness(X, y, sensitive_col_idx):
    unsensitive_x, _ = _separate_sensitive_attrs(X, sensitive_col_idx)

    alpha = 1.0 # alpha = 1 / C, a regularization parameter
    w = minimize(fun=_logistic_loss_and_grad,
                 x0=np.random.rand(unsensitive_x.shape[1],),
                 args=(unsensitive_x, y, alpha),
                 method='SLSQP',
                 options={"maxiter": 10000},
                 jac=True
                )

    return w.x

class FairLogitEstimator(BaseEstimator, ClassifierMixin):
    """ A logistic regression estimator that also takes into account fairness
        over sensitive attributes
    """

    def fit(self, X, y, sensitive_col_idx, correlation_tolerance=None):
        """A reference implementation of a fitting function
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        sensitive_col_idx : array-like, shape = [n_sensitive attrs]
            Specifies which column(s) of X contain(s) the sensitive
            attribute.
        correlation_tolerance : array-like, optional, shape = [n_sensitive attrs]
            Threshhold below which the correlation should be constrained
            for each sensitive attr. If unspecified, will be 0.2 for
            all sensitive attrs.
        Returns
        -------
        self : object
            Returns self.
        """
        # check if X & y have the correct shape
        X, y = check_X_y(X, y, y_numeric=True)
        sensitive_col_idx = np.reshape(sensitive_col_idx, -1)
        correlation_tolerance = np.reshape(correlation_tolerance, -1)
        if not sensitive_col_idx.shape == correlation_tolerance.shape:
            raise ValueError("Sensitive column indices & correlation tolerances "
                             "have different shapes.")
        # Store the classes seen during fit
        # self.classes_ = unique_labels(y)
        if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise ValueError("Training data has non-numeric dtype. X: %s y: %s"
                             % (X.dtype, y.dtype))

        if len(np.unique(y)) > 2:
            raise ValueError("Only two y values are permissible for a binary logit classifier.")

        if correlation_tolerance is None:
            correlation_tolerance = 0.1*np.ones(sensitive_col_idx.shape[0])

        self.sensitive_col_idx_ = sensitive_col_idx

        self.w_ = _train_model_for_fairness(X, y, sensitive_col_idx,
                                            correlation_tolerance)

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
        X = np.delete(X, self.sensitive_col_idx_, 1)

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
        X = np.delete(X, self.sensitive_col_idx_, 1)

        log_probs = np.dot(X, self.w_)
        probs = 1./(1 + np.exp(-log_probs))

        return np.array([1 - probs, probs])

    def boundary_distances(self, X):
        """ Returns the dot product of each sample in X
        with w.
        """
        check_is_fitted(self, ['w_'])
        X = check_array(X)

        # remove sensitive col from test input
        X = np.delete(X, self.sensitive_col_idx_, 1)

        return np.dot(X, self.w_)
