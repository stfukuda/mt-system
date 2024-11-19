"""
MT, MTA and Standardized Variation Pressure Methods Module.
"""

# Authors: Shota Fukuda <st_fukuda@outlook.jp>
# License: BSD-3-Clause

from numbers import Integral, Real

import numpy as np
from scipy.linalg import pinvh
from scipy.stats import chi2, f
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted


class MT(BaseEstimator):
    """
    MT, MTA and Standardized-Variation-Pressure methods.

    The MT, MTA and SVP methods are unsupervised learning methods used for
    pattern recognition in quality engineering. These methods learn the mean
    and standard deviation of each feature and the inverse correlation
    matrix of the training data, and compute MD values based on these
    values. The training data is called the unit space and usually contains
    only normal data. The MTA method learns an adjoint matrix instead of an
    inverse matrix to deal with multicolinearity. The SVP method does not
    require a correlation matrix.
    """

    _parameter_constraints: dict = {
        "method": [StrOptions({"mt", "mta", "svp"})],
        "ddof": [Interval(Integral, 0, None, closed="left")],
        "esp": [Interval(Real, 0, None, closed="right")],
        "kind": [StrOptions({"k", "f", "chi2", "specify"})],
        "a": [Interval(Real, 0, None, closed="right")],
        "threshold": [Interval(Real, 0, None, closed="right")],
        "return_sqrt": ["boolean"],
    }

    def __init__(
        self,
        *,
        method: str = "mt",
        ddof: int = 1,
        esp: float = 1e-16,
        kind: str = "specify",
        a: float = 0.05,
        threshold: float = 4.0,
        return_sqrt: bool = False,
    ):
        """
        Initialize the instance.

        Parameters
        ----------
        method : {"mt", "mta", "svp"}, default="mt"
            Computation method.

        ddof : int, default=1
            It means the delta degrees of freedom. The divisor used in the is
            `N - ddof`, where `N` is the number of samples.

        esp : float, default=1e-16
            A constant to avoid zero division. It is used in the calculation as
            `1 / (x + esp)`.

        kind : {"k", "f", "chi2", "specify"}, default="specify"
            The distribution used to determine normal and abnormal thresholds.

        a : float, default=0.05
            Right side significance level. Use to set the threshold when type is
            set to `f` or `chi2`.

        threshold : float, default=4.0
            Threshold to use when `kind` is set to `specify`.

        return_sqrt : bool, default=False
            Return the square root of the MD value or not.

        Attributes
        ----------
        mean_ : ndarray of shape (n_features, )
            Means of each feature of the training data.

        scale_ : ndarray of shape (n_features, )
            Standard deviation values of each feature of the training data.

        covariance_ : ndarray of shape (n_features, n_features)
            Correlation matrix, variance-covariance matrix, or identity matrix
            of the training data; correlation matrix if "method" is "mt",
            variance-covariance matrix if "method" is "mta", or identity matrix
            if "method" is "svp".

        precision_ : ndarray of shape (n_features, n_features)
            The inverse matrix or adjoint matrix of covariance_; if method is
            svp, then identity matrix.

        dist_ : ndarray of shape(n_samples, )
            Mahalanobis distances of the training set (on which the fit is
            called) observations.

        n_features_in_ : int
            Number of features seen during fit.

        feature_names_in_ : ndarray of shape (n_features_in_, )
            Names of features seen during the fit. Defined only if X has feature
            names that are all strings.
        """
        self.method = method
        self.ddof = ddof
        self.esp = esp
        self.kind = kind
        self.a = a
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
            Ignore

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

        n, k = X.shape  # type: ignore

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, ddof=self.ddof, axis=0)

        if self.method == "mt":
            std_X = (X - self.mean_[None, :]) / (self.scale_[None, :] + self.esp)
            self.covariance_ = np.corrcoef(std_X, rowvar=False)
        elif self.method == "mta":
            std_X = X - self.mean_[None, :]
            self.covariance_ = np.cov(std_X, rowvar=False)
        else:
            self.covariance_ = np.eye(k)

        self.precision_ = self._get_precision(self.covariance_)

        self.dist_ = self._mahalanobis(X, self.mean_, self.scale_, self.precision_)

        if self.kind == "k":
            self.threshold_ = 4 * k
        elif self.kind == "f":
            self.threshold_ = (
                (k * (n - 1) * (n + 1)) / (n * (n - k)) * f.isf(self.a, k, n - k)
            )
        elif self.kind == "chi2":
            self.threshold_ = chi2.isf(self.a, k)
        else:
            self.threshold_ = self.threshold

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

        return np.where(self.mahalanobis(X=X) >= self.threshold_, 1, 0)

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

        MD = self._mahalanobis(X, self.mean_, self.scale_, self.precision_)

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

    def _cofactor(self, X: np.ndarray, i: int, j: int):
        """
        Compute the (i, j) cofactor of X.

        Parameters
        ----------
        X : ndarray of shape (n_features, n_features)
            Covariance matrix.

        i : int
            Cofactor row number. Start from 0.

        j : int
            Cofactor column number. Start from 0.

        Returns
        -------
        cofactor : float
            (i, j) cofactor of X.
        """
        M = np.delete(X, i, axis=0)
        M = np.delete(M, j, axis=1)

        return (-1) ** (i + j) * np.linalg.det(M)

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
            The inverse or adjoint matrix of the covariance matrix.
        """
        if self.method == "mt":
            precision = pinvh(covariance, check_finite=False)
        elif self.method == "mta":
            m = covariance.shape[0]

            precision = np.empty_like(covariance)

            for i in range(m):
                for j in range(m):
                    precision[j, i] = self._cofactor(covariance, i, j)
        else:
            m = covariance.shape[0]
            precision = np.eye(m)

        return precision

    def _mahalanobis(self, X, mean, scale, precision):
        """
        Compute Mahalanobis distances (MD values).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        mean : ndarray of shape (n_features, )
            Mean values of the unit space.

        scale : ndarray of shape (n_features, )
            Standard deviation of the unit space.

        precision : ndarray of shape (n_features, n_features)
            Correlation matrix inverse of the unit space.

        Returns
        -------
        MD : ndarray of shape (n_samples, )
            Mahalanobis distances (MD values).
        """

        if self.method in ["mt", "svp"]:
            std_X = (X - mean[None, :]) / (scale[None, :] + self.esp)
        else:
            std_X = X - mean[None, :]

        MD = (std_X.dot(precision) * std_X).mean(axis=1)

        if self.return_sqrt:
            MD = np.sqrt(MD)

        return MD

    def _more_tags(self) -> dict:
        return {"binary_only": True}
