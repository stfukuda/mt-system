# Authors: Shota Fukuda <st_fukuda@outlook.jp>
# License: BSD-3-Clause

from sklearn.utils.estimator_checks import parametrize_with_checks

from mts._msr import MSR


@parametrize_with_checks([MSR()])
def test_check_estimator(estimator, check):
    check(estimator)
