# Authors: Shota Fukuda <st_fukuda@outlook.jp>
# License: BSD-3-Clause

from sklearn.utils.estimator_checks import parametrize_with_checks

from mts import RT


@parametrize_with_checks([RT(return_sqrt=False), RT(return_sqrt=True)])
def test_check_estimator(estimator, check):
    check(estimator)
