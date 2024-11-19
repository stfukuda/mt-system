"""
RT Method Module.
"""

# Authors: Shota Fukuda <st_fukuda@outlook.jp>
# License: BSD-3-Clause

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted


class RT(BaseEstimator):
    """
    RT method.

    The RT method is an unsupervised learning method used for pattern
    recognition in quality engineering. The method learns the mean of each
    feature in unit space, the sensitivity and SN ratio of each sample, and
    the associated covariance matrix of the sensitivity and SN ratio, and
    computes MD values based on these values.
    """

    _parameter_constraints: dict = {
        "ddof": [Interval(Integral, 0, None, closed="left")],
        "esp": [Interval(Real, 0, None, closed="right")],
        "threshold": [Interval(Real, 0, None, closed="right")],
        "return_sqrt": ["boolean"],
    }

    def __init__(
        self,
        *,
        ddof: int = 1,
        esp: float = 1e-16,
        threshold: float = 4.0,
        return_sqrt: bool = False,
    ):
        """
        Initialize the instance.

        Parameters
        ----------
        ddof : int, default=1
            It means the delta degrees of freedom. The divisor used in the is
            `N - ddof`, where `N` is the number of samples.

        esp : float, default=1e-16
            A constant to avoid zero division. It is used in the calculation as
            `1 / (x + esp)`.

        threshold : float, default=4.0
            Threshold. A multiple of the standard deviation of the MD values in
            the unit space. If 4, threshold is 4 sigma.

        return_sqrt : bool, default=False
            Return the square root of the MD values or not.

        Attributes
        ----------
        mean_X_ : ndarray of shape (n_features, )
            Mean values of each feature of the training data.

        mean_Y_ : ndarray of shape (2, )
            Means of sensitivity and error variance reciprocals. Mean_Y_[0]` is
            the sensitivity mean, and Mean_Y_[1]` is the error variance
            reciprocal.

        covariance_ : ndarray of shape (2, 2)
            Variance-covariance matrix of sensitivity and error variance
            reciprocal.

        precision_ : ndarray of shape (2, 2)
            Adjoint matrix of `covariance_`.

        dist_ : ndarray of shape(n_samples, )
            Mahalanobis distances of the training set (on which the fit is
            called) observations.

        n_features_in_ : int
            Number of features seen during fit.

        feature_names_in_ : ndarray of shape (n_features_in_, )
            Names of features seen during the fit. Defined only if X has feature
            names that are all strings.
        """
        self.ddof = ddof
        self.esp = esp
        self.threshold = threshold
        self.return_sqrt = return_sqrt

    def fit(self, X, y=None):
        """
        Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted model.
        """
        self._validate_params()  # type: ignore

        X = self._validate_data(  # type: ignore
            X=X,
            reset=True,
            ensure_min_samples=2,
            ensure_min_features=2,
            estimator=self,
        )

        self.mean_X_ = np.mean(X, axis=0)

        Y = self._compute_Y(X, self.mean_X_)

        self.mean_Y_ = np.mean(Y, axis=0)

        std_Y = Y - self.mean_Y_[None, :]

        self.covariance_ = np.cov(std_Y, rowvar=False, ddof=self.ddof)

        self.precision_ = self._get_precision(self.covariance_)

        self.dist_ = self._mahalanobis(Y, self.mean_Y_, self.precision_)

        self.sigma_ = np.sqrt(np.mean(self.dist_))

        return self

    def predict(self, X, y=None):
        """
        Predict the labels of X according to the fitted model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        y : None
            Ignored.

        Returns
        -------
        labels : ndarray of shape (n_samples, )
            Returns 1 for anomalies/outliers and 0 for inliers.
        """
        check_is_fitted(self)

        threshold = self.threshold * self.sigma_

        return np.where(self.mahalanobis(X=X) >= threshold, 1, 0)

    def fit_predict(self, X, y=None):
        """
        Perform Fit to X and Return Labels for X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : None
            Ignored.

        Returns
        -------
        labels : ndarray of shape (n_samples, )
            Returns 1 for anomalies/outliers and 0 for inliers.
        """
        return self.fit(X).predict(X)

    def mahalanobis(self, X):
        """
        Compute the Mahalanobis distances (MD values).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        MD : ndarray of shape (n_samples, )
            Mahalanobis distances (MD values).
        """
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)  # type: ignore

        Y = self._compute_Y(X, self.mean_X_)

        MD = self._mahalanobis(Y, self.mean_Y_, self.precision_)

        if self.return_sqrt:
            MD = np.sqrt(MD)

        return MD

    def score(self, X, y):
        """
        Return the ROCAUC to the given test data and labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.

        y : ndarray of shape (n_samples, )
            True labels for X. 1 for anomalies/outliers and 0 for inliers.

        Returns
        -------
        score : float
            ROCAUC.
        """
        check_is_fitted(self)

        X, y = self._validate_data(X=X, y=y, reset=False)  # type: ignore

        return roc_auc_score(y, self.mahalanobis(X=X))

    def score_samples(self, X):
        """
        Compute the Mahalanobis distances (MD values).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        MD : ndarray of shape (n_samples, )
            MD values.
        """
        check_is_fitted(self)

        return self.mahalanobis(X=X)

    def _compute_Y(self, X, mean_X):
        """
        Compute the sensitivity and SN ratio as Y.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        mean_X : ndarray of shape (n_features, )
            Mean values of the unit space.

        Returns
        -------
        Y : ndarray of shape (n_samples, 2)
            Sensitivity and SN Ratio. Where Y[i, 0] is the sensitivity of the
            i-th sample and Y[i, 1] is the Y[i, 1] is the SN ratio of the i-th
            sample.
        """
        std_X = X - mean_X[None, :]

        r = np.sum(mean_X**2)
        L = np.dot(std_X, mean_X)
        st = np.sum(std_X**2, axis=1)
        sb = (L**2) / (r + self.esp)
        se = st - sb
        ve = se / (std_X.shape[1] - 1)
        y1 = L / (r + self.esp)
        y2 = np.sqrt(ve)
        Y = np.hstack([y1[:, None], y2[:, None]])

        return Y

    def _get_precision(self, covariance):
        """
        Compute precision from covariance.

        Parameters
        ----------
        covariance : ndarray of shape (n_features, n_features)
            Covariance matorix of the unit space.

        Returns
        -------
        precision : ndarray of shape (n_features, n_features)
            Adjoint matrix of the covariance matrix.
        """
        precision = np.empty((2, 2))
        precision[0, 0] = covariance[1, 1]
        precision[0, 1] = -covariance[1, 0]
        precision[1, 0] = -covariance[0, 1]
        precision[1, 1] = covariance[0, 0]

        return precision

    def _mahalanobis(self, Y, mean_Y, precision):
        """
        Compute Mahalanobis distances (MD values).

        Parameters
        ----------
        Y : ndarray of shape (n_samples, 2)
            The sensitivity and the SN ratio of the samples.

        mean_Y : ndarray of shape (2, )
            Mean values of Y in the unit space.

        precision : ndarray of shape (2, 2), default=None
            Adjoint matrix of the covariance matrix in unti space.

        Returns
        -------
        MD : ndarray of shape (n_samples, )
            Mahalanobis distances (MD values).
        """
        std_Y = Y - mean_Y[None, :]

        MD = (std_Y.dot(precision) * std_Y).mean(axis=1)

        if self.return_sqrt:
            MD = np.sqrt(MD)

        return MD

    def _more_tags(self):
        return {"binary_only": True}
