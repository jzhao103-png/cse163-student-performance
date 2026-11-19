"""Statistical tests and effect estimates for the three research questions."""

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
    """Return grade arrays in order from shortest to longest study time."""
    return [
        sample.loc[sample["studytime"] == code, "G3"].to_numpy()
        for code in STUDY_CODES
    ]


def failure_grades(sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return grades for students with no failures, then with any failure."""
    none = sample.loc[sample["failures"] == 0, "G3"].to_numpy()
    any_failure = sample.loc[sample["failures"] > 0, "G3"].to_numpy()
    return none, any_failure


def median_difference(first: np.ndarray, second: np.ndarray) -> float:
    """Return the first sample's median minus the second sample's median."""
    return float(np.median(first) - np.median(second))


def mean_difference(first: np.ndarray, second: np.ndarray) -> float:
    """Return the first sample's mean minus the second sample's mean."""
    return float(np.mean(first) - np.mean(second))


def spearman_coefficient(absences: np.ndarray, grades: np.ndarray) -> float:
    """Return Spearman's rho for paired absence and grade observations."""
    return float(stats.spearmanr(absences, grades).statistic)


def bootstrap_interval(
    samples: tuple[np.ndarray, np.ndarray],
    effect: Callable[[np.ndarray, np.ndarray], float],
    paired: bool = False,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Return a seeded percentile bootstrap 95% interval for an effect."""
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
    """Return a p-value adjusted for a family of three comparisons."""
    return min(NUMBER_OF_TESTS * p_value, 1.0)


def adjacent_comparisons(
    groups: list[np.ndarray], overall_p: float
) -> pd.DataFrame:
    """Test adjacent study groups if the adjusted omnibus test rejects."""
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
    """Run the three planned tests and estimate their effect intervals."""
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
    """Return the same three effect estimates for a sensitivity subset."""
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
