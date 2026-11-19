"""Tests using small examples with hand-checked expected outputs."""

from io import StringIO
from math import isclose, sqrt

import numpy as np
import pandas as pd

from eda import (
    DATA_PATH,
    absence_grade_counts,
    audit_data,
    category_counts,
    grade_summary,
    load_data,
    prepare_data,
    quantitative_summary,
    question_samples,
)
from inference import (
    adjacent_comparisons,
    bonferroni,
    bootstrap_interval,
    effect_estimates,
    mean_difference,
)


def test_read_delimiter_and_missing_values() -> None:
    """Read semicolons and missing markers without losing a zero grade."""
    file = StringIO(
        'school;studytime;absences;failures;G3\n'
        'GP;1;0;0;0\n'
        'MS;2;?;1;12\n'
        'GP;3;1;0;"  "\n'
    )
    raw = load_data(file)
    assert raw.shape == (3, 5)
    assert prepare_data(raw).loc[0, "G3"] == 0
    assert pd.isna(raw.loc[1, "absences"])
    assert pd.isna(raw.loc[2, "G3"])


def test_filter_only_relevant_values() -> None:
    """Keep valid zeros and reject only the relevant invalid values."""
    raw = pd.DataFrame(
        {
            "school": ["GP", "MS", "GP", "GP", "MS", "GP"],
            "studytime": [1, 2, 5, 3, 4, 4],
            "absences": [0, -1, 7, 4, 32, 1.5],
            "failures": [0, 2, 3, -1, 0, 1],
            "G3": [0, 15, 12, 13, 20, 21],
        }
    )
    data = prepare_data(raw)
    assert len(data) == len(raw)
    assert data["failure_group"].iloc[:3].tolist() == [
        "None", "At least one", "At least one"
    ]
    assert audit_data(raw, data)["invalid_analysis_cells"] == 5
    samples = question_samples(data)
    assert samples["Study time"].index.tolist() == [0, 1, 3, 4]
    assert samples["Absences"].index.tolist() == [0, 2, 3, 4]
    assert samples["Prior failures"].index.tolist() == [0, 1, 2, 4]


def test_duplicate_profiles_and_rows() -> None:
    """Count full copies separately from matching analysis profiles."""
    raw = pd.DataFrame(
        {
            "school": ["GP"] * 3, "studytime": [1] * 3,
            "absences": [2] * 3, "failures": [0] * 3,
            "G3": [14] * 3, "sex": ["F", "M", "F"],
        }
    )
    data = prepare_data(raw)
    audit = audit_data(raw, data)
    assert audit["extra_full_duplicates"] == 1
    assert audit["extra_matching_profiles"] == 2
    assert len(data) == 3


def test_quantitative_and_group_summaries() -> None:
    """Check the seven statistics, group counts, and quartile ranges."""
    data = pd.DataFrame(
        {"G3": [0, 2, 4, 6, 8], "absences": [0, 2, 2, 3, 9],
         "studytime": [1, 1, 1, 1, 1], "failures": [0, 0, 1, 2, 3],
         "school": ["GP"] * 5}
    )
    data = prepare_data(data)
    grades = quantitative_summary(data).loc["G3"]
    assert list(grades.index) == [
        "mean", "std", "min", "25%", "50%", "75%", "max"
    ]
    assert grades.drop("std").tolist() == [4, 0, 2, 4, 6, 8]
    assert isclose(grades["std"], sqrt(10))
    assert grade_summary(data, "studytime").loc[1, "IQR"] == 4
    counts = category_counts(data)
    assert counts.loc[counts["variable"] == "studytime", "n"].tolist() == [
        5, 0, 0, 0
    ]


def test_absence_grade_counts() -> None:
    """Aggregate only identical pairs and account for every student."""
    sample = pd.DataFrame({"absences": [0, 0, 0, 2], "G3": [4, 4, 10, 10]})
    counts = absence_grade_counts(sample)
    assert counts.values.tolist() == [[0, 4, 2], [0, 10, 1], [2, 10, 1]]
    assert counts["n"].sum() == len(sample)


def test_effects_use_the_intended_directions() -> None:
    """Compare the effects with simple known differences and ranks."""
    raw = pd.DataFrame(
        {"studytime": [1, 2, 3, 4], "absences": [3, 2, 1, 0],
         "G3": [6, 8, 10, 12], "failures": [1, 1, 0, 0],
         "school": ["GP"] * 4}
    )
    effects = effect_estimates(question_samples(prepare_data(raw)))
    assert effects["study_median_gap"] == 6
    assert isclose(effects["absence_rho"], -1)
    assert effects["failure_mean_gap"] == 4


def test_follow_up_gate_and_correction() -> None:
    """Check that only the planned comparisons run and adjust p-values."""
    groups = [np.arange(i, i + 5) for i in range(4)]
    assert adjacent_comparisons(groups, 0.02).empty
    followups = adjacent_comparisons(groups, 0.01)
    pairs = followups[["first_code", "second_code"]].values.tolist()
    assert pairs == [[1, 2], [2, 3], [3, 4]]
    assert isclose(bonferroni(0.01), 0.03)
    assert bonferroni(0.5) == 1


def test_paired_bootstrap() -> None:
    """Resampling paired values must preserve a fixed difference."""
    first = np.array([1, 3, 7, 9])
    second = first - 1
    interval = bootstrap_interval(
        (first, second), mean_difference, paired=True, resamples=250
    )
    assert interval == (1.0, 1.0)


def test_shipped_data() -> None:
    """Check the expected source shape and the analysis sample sizes."""
    raw = load_data(DATA_PATH)
    data = prepare_data(raw)
    audit = audit_data(raw, data)
    assert raw.shape == (649, 33)
    assert audit["missing_cells"] == audit["invalid_analysis_cells"] == 0
    assert audit["extra_full_duplicates"] == 0
    for sample in question_samples(data).values():
        assert len(sample) == len(raw)


def main() -> None:
    """Run the data preparation, analysis, and source data checks."""
    tests = [
        test_read_delimiter_and_missing_values,
        test_filter_only_relevant_values,
        test_duplicate_profiles_and_rows,
        test_quantitative_and_group_summaries,
        test_absence_grade_counts,
        test_effects_use_the_intended_directions,
        test_follow_up_gate_and_correction,
        test_paired_bootstrap,
        test_shipped_data,
    ]
    for test in tests:
        test()
    print(f"Passed {len(tests)} EDA tests.")


if __name__ == "__main__":
    main()
