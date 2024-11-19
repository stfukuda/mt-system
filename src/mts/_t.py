"""
T(1), T(2), Ta and Tb Methods Module.
"""

# Authors: Shota Fukuda <st_fukuda@outlook.jp>
# License: BSD-3-Clause

from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted


class T(RegressorMixin, BaseEstimator):
    """
    T(1), T(2), Ta and Tb methods.

    The T(1), T(2), Ta and Tb methods are supervised learning methods used
    for regression in quality engineering. The T(1) and T(2) methods divide
    the training data into unit space and signal data, and learn the mean
    from the unit space and the sensitivity and SN ratio from the signal
    data. The Ta method does not divide the training data into unit space
    and signal data, and learns the mean, sensitivity, and SN ratio from all
    the training data. The Tb method also learns from all training data, but
    for each element, the sample with the largest SN ratio is used as the
    mean.
    """

    _parameter_constraints: dict = {
        "tb": ["boolean"],
        "esp": [Interval(Real, 0, None, closed="right")],
        "is_simplified": ["boolean"],
    }

    def __init__(
        self, *, tb: bool = False, esp: float = 1e-16, is_simplified: bool = False
    ):
        """
        Initialize the instance.

        Parameters
        ----------
        tb : bool, default=False
            Whether to compute as Tb method. If False, compute as T(1), T(2),
            and Ta methods.

        esp : float, default=1e-16
            A constant to avoid zero division. It is used in the calculation as
            `1 / (x + esp)`.

        is_simplified : bool, default=False
            Compute the SN ratio using the simplified formula or not. The
            simplified formula computes with `b**2 / ve`.

        Attributes
        ----------
        mean_X_ : ndarray of shape (n_features, )
            Mean values of each feature of the training data.

        mean_y_ : float or ndarray of shape (n_features, )
            Mean value of target values.

        n_ : ndarray of shape (n_features, )
            SN ratio between each feature and the target values.

        b_ : ndarray of shape (n_features, )
            Sensitivity between each feature and target values.

        n_features_in_ : int
            Number of features seen during fit.

        feature_names_in_ : ndarray of shape (n_features_in_, )
            Names of features seen during the fit. Defined only if X has feature
            names that are all strings.
        """
        self.tb = tb
        self.esp = esp
        self.is_simplified = is_simplified

    def fit(self, X, y, *, us_idx=None):
        """
        Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data. Includes unit space and signal data.

        y : ndarray of shape (n_samples, )
            Target values. Will be cast to X's dtype if necessary.

        us_idx : array_like of shape (n_samples, ) or None, default=None.
            A binary array indicating which sample of the training data is the
            unit space (0 for the unit space, 1 for the signal data); if None,
            the training data is not divided into the unit space and the signal
            data, but is computed as the Ta method. It is ignored when the Tb
            method is computed.

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
            estimator=self,
        )

        if self.tb:
            n = np.empty_like(X)
            b = np.empty_like(X)
            for i, (x_i, y_i) in enumerate(zip(X, y)):
                std_X = X - x_i[None, :]
                std_y = y - y_i

                n[i], b[i] = self._compute_sn_ratio_and_sensitivity(std_X, std_y)

            idx_row = np.argmax(n, axis=0)
            idx_col = np.arange(X.shape[1])

            self.mean_X_ = X[idx_row, idx_col]
            self.mean_y_ = y[idx_row]

            self.b_ = b[idx_row, idx_col]
            self.n_ = n[idx_row, idx_col]
        else:
            if us_idx is None:
                self.mean_X_ = np.mean(X, axis=0)
                self.mean_y_ = np.mean(y)

                std_X = X - self.mean_X_[None, :]
                std_y = y - self.mean_y_
            else:
                unit_space_mask = np.where(us_idx == 0, True, False)

                self.mean_X_ = np.mean(X[unit_space_mask], axis=0)
                self.mean_y_ = np.mean(y[unit_space_mask])

                std_X = X[~unit_space_mask] - self.mean_X_[None, :]
                std_y = X[~unit_space_mask] - self.mean_y_

            self.n_, self.b_ = self._compute_sn_ratio_and_sensitivity(std_X, std_y)

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
            Predict values.
        """
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False, estimator=self)  # type: ignore

        std_X = X - self.mean_X_

        M_pred = std_X / (self.b_ + self.esp)[None, :]

        if self.tb:
            y_pred = M_pred + self.mean_y_[None, :]  # type: ignore
            y_pred = np.dot(y_pred, self.n_) / (np.sum(self.n_) + self.esp)
        else:
            M_pred = np.dot(M_pred, self.n_) / (np.sum(self.n_) + self.esp)
            y_pred = M_pred + self.mean_y_

        return y_pred

    def score(self, X, y):
        """
        Return the SN ratio of the integrated estimate.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.

        y : ndarray of shape (n_samples, )
            True values for X.

        Returns
        -------
        n : float
            SN ratio of the integrated estimate. It is computed from M_True and
            M_Pred for the T(1), T(2) and Ta methods, and from y_True and y_Pred
            for the Tb method.
        """
        check_is_fitted(self)

        X, y = self._validate_data(X=X, y=y, reset=False)  # type: ignore

        if self.tb:
            M_true = y
            M_pred = self.predict(X)
        else:
            M_true = y - self.mean_y_
            M_pred = self.predict(X) - self.mean_y_

        n, _ = self._compute_sn_ratio_and_sensitivity(M_pred[:, None], M_true)
        n = 10 * np.log10(n)

        return n

    def _compute_sn_ratio_and_sensitivity(self, X, y):
        """
        Compute the SN ratio and sensitivity.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        y : ndarray of shape (n_samples, )
            Target values.

        Returns
        -------
        n : ndarray of shape (n_features, )
            SN ratio between each feature and the target values.

        b : ndarray of shape (n_features, )
            Sensitivity between each feature and the target values.
        """
        r = np.sum(y**2)
        L = np.dot(X.T, y)
        st = np.sum(X**2, axis=0)
        sb = (L**2) / (r + self.esp)
        se = st - sb
        ve = se / (X.shape[0] - 1)

        b = L / (r + self.esp)

        if self.is_simplified:
            n = (b**2) / (ve + self.esp)
        else:
            n = (1 / r * (sb - ve)) / (ve + self.esp)
            mask = sb <= ve
            n[mask] = 0

        return n, b

    def _more_tags(self):
        return RegressorMixin._more_tags(self)  # type: ignore
