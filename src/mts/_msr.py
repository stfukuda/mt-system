"""
MSR Method Module.
"""

# Authors: Shota Fukuda <st_fukuda@outlook.jp>
# License: BSD-3-Clause

from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted


class MSR(RegressorMixin, BaseEstimator):
    """
    MSR: Multiple Single Regression.
    """

    _parameter_constraints: dict = {
        "delta": [Interval(Real, 0, None, closed="right")],
        "esp": [Interval(Real, 0, None, closed="right")],
    }

    def __init__(self, *, delta: float = 1e-4, esp: float = 1e-16):
        """
        Initialize the instance.

        Parameters
        ----------
        delta : float, default=1e-4
            Threshold for stopping repeated computations.

        esp : float, default=1e-16
            A constant to avoid zero division. It is used in the calculation as
            `1 / (x + esp)`.

        Attributes
        ----------
        mean_X_ : ndarray of shape(n_features, )
            Mean values of each feature of the training data.

        mean_y_ : float
            Mean value of target values.

        coef_ : ndarray of shape (n_features, )
            Estimated coefficients for the MSR.

        n_features_in_ : int
            Number of features seen during fit.

        feature_names_in_ : ndarray of shape (n_features_in_, )
            Names of features seen during the fit. Defined only if X has feature
            names that are all strings.

        References
        ----------
        前田誠. (2017). T 法 (1) の考え方を利用した新しい回帰手法の提案. 品質, 47(2),
        185-194.
        """
        self.delta = delta
        self.esp = esp

    def _compute_sn_ratio_and_sensitivity(self, X, y):
        r = np.sum(X**2, axis=0)
        L = np.dot(X.T, y)
        st = np.dot(y, y)
        sb = (L**2) / (r + self.esp)
        se = st - sb
        n = np.sqrt(sb / (se + self.esp))
        b = L / (r + self.esp)

        return st, sb, n, b

    def fit(self, X, y):
        """
        Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples, )
            Target values. Will be cast to X's dtype if necessary.

        Returns
        -------
        self : object
            Fitted model.
        """
        self._validate_params()  # type: ignore

        X, y = self._validate_data(  # type: ignore
            X=X,
            y=y,
            reset=True,
            y_numeric=True,
            ensure_min_samples=2,
            estimator=self,
        )

        n_samples, n_features = X.shape

        if n_samples <= 50:
            n_splits = n_samples
        else:
            n_splits = int(2250 / n_samples) + 5

        kf = KFold(n_splits=n_splits)

        self.coef_ = np.zeros(n_features)
        coef_kf = np.zeros((n_splits, n_features))

        self.mean_X_ = np.mean(X, axis=0)
        self.mean_y_ = np.mean(y)

        std_X = X - self.mean_X_[None, :]
        std_y = y - self.mean_y_

        zz_before = None
        skip_kf = []
        while True:
            y_ = np.dot(std_X, self.coef_)

            z = std_y - y_

            st, sb, n, b = self._compute_sn_ratio_and_sensitivity(std_X, z)

            if st == 0 or np.all(sb == 0):
                break

            self.coef_ += b * n / np.sum(n)

            z = np.empty(n_samples)
            for kf_idx, (train_idx, test_idx) in enumerate(kf.split(std_X)):
                train_X, train_y = std_X[train_idx], std_y[train_idx]
                test_X, test_y = std_X[test_idx], std_y[test_idx]

                if kf_idx in skip_kf:
                    y_kf = np.dot(test_X, coef_kf[kf_idx])
                    z[test_idx] = test_y - y_kf
                    continue

                y_kf = np.dot(train_X, coef_kf[kf_idx])
                z_kf = train_y - y_kf

                st, sb, n, b = self._compute_sn_ratio_and_sensitivity(train_X, z_kf)

                if st == 0 or np.all(sb == 0):
                    skip_kf.append(kf_idx)
                    y_kf = np.dot(test_X, coef_kf[kf_idx])
                    z[test_idx] = test_y - y_kf
                    continue

                coef_kf[kf_idx] += b * n / np.sum(n)

                y_kf = np.dot(test_X, coef_kf[kf_idx])
                z[test_idx] = test_y - y_kf

            zz_after = np.dot(z, z)

            if zz_before is None:
                zz_before = zz_after * 2

            if (zz_before - zz_after) <= (self.delta * zz_before):
                break
            else:
                zz_before = zz_after

        return self

    def predict(self, X, y=None):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        y : None
            Ignored.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, )
            Predicted values.
        """
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)  # type: ignore

        std_X = X - self.mean_X_[None, :]

        return np.dot(std_X, self.coef_) + self.mean_y_

    def _more_tags(self):
        return RegressorMixin._more_tags(self)  # type: ignore
