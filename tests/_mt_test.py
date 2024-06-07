# Authors: Shota Fukuda <st_fukuda@outlook.jp>
# License: BSD-3-Clause

from sklearn.utils.estimator_checks import parametrize_with_checks

from mts import MT


@parametrize_with_checks(
    [
        MT(method="mt", kind="k", return_sqrt=False),
        MT(method="mt", kind="k", return_sqrt=True),
        MT(method="mt", kind="f", return_sqrt=False),
        MT(method="mt", kind="f", return_sqrt=True),
        MT(method="mt", kind="chi2", return_sqrt=False),
        MT(method="mt", kind="chi2", return_sqrt=True),
        MT(method="mt", kind="specify", return_sqrt=False),
        MT(method="mt", kind="specify", return_sqrt=True),
        MT(method="mta", kind="k", return_sqrt=False),
        MT(method="mta", kind="k", return_sqrt=True),
        MT(method="mta", kind="f", return_sqrt=False),
        MT(method="mta", kind="f", return_sqrt=True),
        MT(method="mta", kind="chi2", return_sqrt=False),
        MT(method="mta", kind="chi2", return_sqrt=True),
        MT(method="mta", kind="specify", return_sqrt=False),
        MT(method="mta", kind="specify", return_sqrt=True),
        MT(method="svp", kind="k", return_sqrt=False),
        MT(method="svp", kind="k", return_sqrt=True),
        MT(method="svp", kind="f", return_sqrt=False),
        MT(method="svp", kind="f", return_sqrt=True),
        MT(method="svp", kind="chi2", return_sqrt=False),
        MT(method="svp", kind="chi2", return_sqrt=True),
        MT(method="svp", kind="specify", return_sqrt=False),
        MT(method="svp", kind="specify", return_sqrt=True),
    ]
)
def test_check_estimator(estimator, check):
    check(estimator)
