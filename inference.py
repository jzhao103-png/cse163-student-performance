"""
Junyi Zhao
CSE 163 AF, Autumn 2026 - Final Project Part 2 (EDA)

Statistical inference for the three research questions with
scipy.stats: a Kruskal-Wallis test with adjacent Mann-Whitney U
follow-ups (study time), a Spearman correlation (absences), and a Welch
t-test (prior failures), each with a Bonferroni adjustment and a seeded
percentile bootstrap 95% interval for its effect.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats


STUDY_CODES = [1, 2, 3, 4]
FAMILY_ALPHA = 0.05
NUMBER_OF_TESTS = 3
TEST_ALPHA = FAMILY_ALPHA / NUMBER_OF_TESTS
BOOTSTRAP_SEED = 163
BOOTSTRAP_RESAMPLES = 5000


def study_grades(sample: pd.DataFrame) -> list[np.ndarray]:
    """
    Returns a list of four arrays with the G3 values in sample for
    study-time codes 1, 2, 3, and 4, in that order.
    """
    return [
        sample.loc[sample["studytime"] == code, "G3"].to_numpy()
        for code in STUDY_CODES
    ]


def failure_grades(sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a tuple of two arrays from sample: the G3 values of students
    with no prior failures, then those of students with at least one.
    """
    none = sample.loc[sample["failures"] == 0, "G3"].to_numpy()
    any_failure = sample.loc[sample["failures"] > 0, "G3"].to_numpy()
    return none, any_failure


def median_difference(first: np.ndarray, second: np.ndarray) -> float:
    """Returns the median of first minus the median of second."""
    return float(np.median(first) - np.median(second))


def mean_difference(first: np.ndarray, second: np.ndarray) -> float:
    """Returns the mean of first minus the mean of second."""
    return float(np.mean(first) - np.mean(second))


def spearman_coefficient(absences: np.ndarray, grades: np.ndarray) -> float:
    """
    Returns Spearman's rank correlation (rho) between the paired absences
    and grades arrays.
    """
    return float(stats.spearmanr(absences, grades).statistic)


def bootstrap_interval(
    samples: tuple[np.ndarray, np.ndarray],
    effect: Callable[[np.ndarray, np.ndarray], float],
    paired: bool = False,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """
    Returns the (low, high) bounds of a seeded 95% percentile bootstrap
    interval for effect(*samples) using the given number of resamples.
    If paired is True, the arrays are resampled together by position so
    pairs stay intact; otherwise each array is resampled on its own.
    """
    interval = stats.bootstrap(
        samples,
        effect,
        vectorized=False,
        paired=paired,
        method="percentile",
        confidence_level=0.95,
        n_resamples=resamples,
        rng=BOOTSTRAP_SEED,
    ).confidence_interval
    return float(interval.low), float(interval.high)


def bonferroni(p_value: float) -> float:
    """
    Returns the Bonferroni-adjusted p-value: p_value times the number of
    planned tests (3), capped at 1.0.
    """
    return min(NUMBER_OF_TESTS * p_value, 1.0)


def adjacent_comparisons(
    groups: list[np.ndarray], overall_p: float
) -> pd.DataFrame:
    """
    Takes the four study-time grade arrays and the omnibus p-value. If
    overall_p is below the Bonferroni threshold (0.05 / 3), returns a
    DataFrame of two-sided Mann-Whitney U tests for codes 1 vs 2, 2 vs 3,
    and 3 vs 4 with raw and adjusted p-values. Otherwise returns an empty
    DataFrame with the same columns.
    """
    columns = ["first_code", "second_code", "U", "p", "adjusted_p"]
    rows = []
    if overall_p < TEST_ALPHA:
        for code in STUDY_CODES[:-1]:
            result = stats.mannwhitneyu(
                groups[code - 1],
                groups[code],
                alternative="two-sided",
                method="asymptotic",
            )
            rows.append(
                [code, code + 1, result.statistic, result.pvalue,
                 bonferroni(result.pvalue)]
            )
    return pd.DataFrame(rows, columns=columns)


def primary_tests(
    samples: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the three planned tests on the per-question samples. Returns a
    tuple of a DataFrame indexed by question (statistic, raw and adjusted
    p-values, rejection decision, effect, and bootstrap interval) and the
    DataFrame of adjacent study-time comparisons.
    """
    study = study_grades(samples["Study time"])
    pairs = samples["Absences"]
    absences = pairs["absences"].to_numpy()
    grades = pairs["G3"].to_numpy()
    none, any_failure = failure_grades(samples["Prior failures"])

    study_test = stats.kruskal(*study)
    absence_test = stats.spearmanr(absences, grades, alternative="two-sided")
    failure_test = stats.ttest_ind(
        none, any_failure, equal_var=False, alternative="two-sided"
    )
    comparisons = adjacent_comparisons(study, study_test.pvalue)

    questions = [
        ("Study time", "Kruskal-Wallis", study_test,
         (study[-1], study[0]), median_difference, False),
        ("Absences", "Spearman", absence_test,
         (absences, grades), spearman_coefficient, True),
        ("Prior failures", "Welch t-test", failure_test,
         (none, any_failure), mean_difference, False),
    ]
    rows = []
    for question, test_name, test, values, effect, paired in questions:
        low, high = bootstrap_interval(values, effect, paired=paired)
        rows.append(
            {
                "question": question,
                "test": test_name,
                "statistic": test.statistic,
                "p": test.pvalue,
                "adjusted_p": bonferroni(test.pvalue),
                "reject": test.pvalue < TEST_ALPHA,
                "effect": effect(*values),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows).set_index("question"), comparisons


def effect_estimates(samples: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Returns a dict with the three effects for the per-question samples:
    the median G3 gap between the longest and shortest study-time bands,
    Spearman's rho for absences and G3, and the mean G3 gap between
    students with no prior failures and those with at least one.
    """
    study = study_grades(samples["Study time"])
    pairs = samples["Absences"]
    none, any_failure = failure_grades(samples["Prior failures"])
    return {
        "study_median_gap": median_difference(study[-1], study[0]),
        "absence_rho": spearman_coefficient(
            pairs["absences"].to_numpy(), pairs["G3"].to_numpy()
        ),
        "failure_mean_gap": mean_difference(none, any_failure),
    }
