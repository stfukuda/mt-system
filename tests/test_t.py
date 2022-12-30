# Authors: Shota Fukuda <st_fukuda@outlook.jp>
# License: BSD-3-Clause

from sklearn.utils.estimator_checks import parametrize_with_checks

from mts import T


@parametrize_with_checks(
    [
        T(tb=False, is_simplified=False),
        T(tb=False, is_simplified=True),
        T(tb=True, is_simplified=False),
        T(tb=True, is_simplified=True),
    ]
)
def test_check_estimator(estimator, check):
    check(estimator)
