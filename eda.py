"""
Junyi Zhao
CSE 163 AF, Autumn 2026 - Final Project Part 2 (EDA)

Runs the exploratory data analysis of the UCI Student Performance
(Portuguese course) data: reads and validates student-por.csv, audits
missing values and duplicates, summarizes the variables of interest,
saves one plot per research question (study time, absences, prior
failures), and prints the statistical tests from inference.py.
Run with `python eda.py`; tables and plots are saved in outputs/.
"""

from pathlib import Path
from typing import TextIO

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from inference import STUDY_CODES, effect_estimates, primary_tests


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "student-por.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
ANALYSIS_COLUMNS = ["school", "studytime", "absences", "failures", "G3"]
SCHOOLS = ["GP", "MS"]
STUDY_LABELS = ["<2 hours", "2–5 hours", "5–10 hours", ">10 hours"]
FAILURE_CODES = [0, 1, 2, 3]
FAILURE_GROUPS = ["None", "At least one"]
QUESTION_COLUMNS = {
    "Study time": ["studytime", "G3"],
    "Absences": ["absences", "G3"],
    "Prior failures": ["failures", "G3"],
}


def load_data(path: Path | TextIO) -> pd.DataFrame:
    """
    Reads the semicolon-delimited UCI CSV at the given path (or an open
    text stream) and returns it as a DataFrame. Entries of "?" and cells
    that are empty or contain only spaces are stored as missing (NA).
    """
    raw = pd.read_csv(path, sep=";", na_values=["?"], skipinitialspace=True)
    return raw.replace(r"^\s*$", pd.NA, regex=True)


def valid_integers(
    values: pd.Series, minimum: int, maximum: int
) -> pd.Series:
    """
    Converts the given Series to numbers and returns a Series that keeps
    only whole numbers from minimum to maximum (inclusive). Any other
    entry (text, decimals, or out-of-range values) becomes NaN.
    """
    numbers = pd.to_numeric(values, errors="coerce")
    is_valid = numbers.between(minimum, maximum) & (numbers % 1 == 0)
    return numbers.where(is_valid)


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw student DataFrame and returns a new DataFrame with the
    five analysis columns, where invalid entries become NaN, plus a
    failure_group column ("None" for 0 failures, "At least one"
    otherwise). No rows are removed. Raises a ValueError if any analysis
    column is missing from raw.
    """
    missing_columns = set(ANALYSIS_COLUMNS) - set(raw.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing columns: {names}")

    data = raw[ANALYSIS_COLUMNS].copy()
    data["school"] = data["school"].where(data["school"].isin(SCHOOLS))
    data["studytime"] = valid_integers(data["studytime"], 1, 4)
    data["absences"] = valid_integers(data["absences"], 0, 93)
    data["failures"] = valid_integers(data["failures"], 0, 3)
    data["G3"] = valid_integers(data["G3"], 0, 20)
    no_failure, any_failure = FAILURE_GROUPS
    data["failure_group"] = data["failures"].map(
        {0: no_failure, 1: any_failure, 2: any_failure, 3: any_failure}
    )
    return data


def audit_data(raw: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
    """
    Takes the raw DataFrame and its prepared version from prepare_data
    and returns a Series of counts: table size, missing cells, invalid
    analysis values, extra full-row duplicates, extra rows with matching
    analysis fields, zero grades, zero absences, and zero grades at zero
    absences.
    """
    missing = raw.isna()
    selected = raw[ANALYSIS_COLUMNS]
    invalid = selected.notna() & data[ANALYSIS_COLUMNS].isna()
    return pd.Series(
        {
            "raw_rows": raw.shape[0],
            "raw_columns": raw.shape[1],
            "missing_cells": missing.sum().sum(),
            "rows_with_missing": missing.any(axis=1).sum(),
            "columns_with_missing": missing.any(axis=0).sum(),
            "invalid_analysis_cells": invalid.sum().sum(),
            "extra_full_duplicates": raw.duplicated().sum(),
            "extra_matching_profiles": selected.duplicated().sum(),
            "zero_final_grades": data["G3"].eq(0).sum(),
            "zero_absences": data["absences"].eq(0).sum(),
            "zero_grades_at_zero_absences": (
                data["G3"].eq(0) & data["absences"].eq(0)
            ).sum(),
        },
        name="count",
    )


def question_samples(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Takes the prepared data and returns a dict that maps each research
    question ("Study time", "Absences", "Prior failures") to the rows
    where both of that question's variables (its predictor and G3) are
    present.
    """
    return {
        question: data.dropna(subset=columns)
        for question, columns in QUESTION_COLUMNS.items()
    }


def quantitative_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row each for G3 and absences in the
    prepared data, giving the mean, standard deviation, minimum, first
    quartile, median, third quartile, and maximum.
    """
    return data[["G3", "absences"]].describe().T.drop(columns="count")


def category_counts(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with the number of students at every level of
    studytime, failures, failure_group, and school in the prepared data.
    Levels that never occur are listed with a count of 0.
    """
    categories = {
        "studytime": STUDY_CODES,
        "failures": FAILURE_CODES,
        "failure_group": FAILURE_GROUPS,
        "school": SCHOOLS,
    }
    counts = []
    for column, levels in categories.items():
        frequencies = data[column].value_counts().reindex(levels, fill_value=0)
        table = pd.DataFrame(
            {"variable": column, "level": levels, "n": frequencies.to_numpy()}
        )
        counts.append(table)
    return pd.concat(counts, ignore_index=True)


def grade_summary(sample: pd.DataFrame, group: str) -> pd.DataFrame:
    """
    Returns the count, mean, standard deviation, minimum, quartiles,
    maximum, and interquartile range (IQR) of G3 for each level of the
    given group column in sample.
    """
    summary = sample.groupby(group)["G3"].describe()
    summary["IQR"] = summary["75%"] - summary["25%"]
    return summary


def absence_grade_counts(sample: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per distinct (absences, G3) pair in
    sample and a column n that counts the students with that pair.
    """
    return sample.groupby(["absences", "G3"]).size().reset_index(name="n")


def sensitivity_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Recomputes the three effect estimates for all records, for students
    with a final grade above zero, and for each school. Returns one row
    per subset with its number of rows and the three effects.
    """
    subsets = {
        "All records": data,
        "G3 above zero": data[data["G3"] > 0],
        "GP only": data[data["school"] == "GP"],
        "MS only": data[data["school"] == "MS"],
    }
    rows = []
    for name, subset in subsets.items():
        samples = question_samples(subset)
        row = {"subset": name, "rows": len(subset)}
        row.update(effect_estimates(samples))
        rows.append(row)
    return pd.DataFrame(rows).set_index("subset")


def save_figure(fig: plt.Figure, filename: Path) -> None:
    """
    Saves the given figure to filename at 220 dpi with a white background
    and then closes the figure.
    """
    fig.tight_layout(pad=0.8)
    fig.savefig(filename, dpi=220, facecolor="white")
    plt.close(fig)


def grade_boxplot(
    data: pd.DataFrame,
    group: str,
    order: list[int] | list[str],
    labels: list[str],
    title: str,
    x_label: str,
) -> plt.Figure:
    """
    Returns a figure with a box plot of G3 for each level of the group
    column in data, drawn in the given order. Each x tick shows the
    matching label and that group's sample size, and title and x_label
    set the plot text.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.18))
    sns.boxplot(
        data=data, x=group, y="G3", order=order, color="#8AC4C0",
        width=0.5, linewidth=1.3, fliersize=3, ax=ax
    )
    counts = data[group].value_counts()
    tick_labels = [
        f"{label}\n(n = {counts.get(level, 0)})"
        for label, level in zip(labels, order)
    ]
    ax.set_xticks(range(len(labels)), labels=tick_labels)
    ax.set(
        title=title, xlabel=x_label, ylabel="Final grade (G3)",
        ylim=(-0.8, 20.8), yticks=[0, 5, 10, 15, 20]
    )
    ax.xaxis.grid(False)
    sns.despine(ax=ax)
    return fig


def absence_plot(sample: pd.DataFrame) -> plt.Figure:
    """
    Returns a figure that plots each distinct (absences, G3) pair in
    sample as a point whose area grows with the number of students who
    share that pair.
    """
    pairs = absence_grade_counts(sample)
    fig, ax = plt.subplots(figsize=(7.2, 3.18))
    sns.scatterplot(
        data=pairs, x="absences", y="G3", size="n", sizes=(14, 148),
        color="#247D83", alpha=0.69, linewidth=0.35, legend="brief", ax=ax
    )
    ax.legend(
        title="Students per pair", loc="upper left",
        bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True
    )
    ax.set(
        title="Final grade and school absences",
        xlabel="Number of absences", ylabel="Final grade (G3)",
        xlim=(-1, sample["absences"].max() + 1), ylim=(-0.8, 20.8),
        yticks=[0, 5, 10, 15, 20]
    )
    sns.despine(ax=ax)
    return fig


def save_plots(samples: dict[str, pd.DataFrame], folder: Path) -> None:
    """
    Saves the three research-question plots (study time, absences, and
    prior failures) as PNG files in folder, using the matching sample
    from samples for each plot.
    """
    sns.set_theme(
        style="whitegrid",
        font="DejaVu Sans",
        rc={"axes.titlesize": 15, "axes.labelsize": 12,
            "xtick.labelsize": 12, "ytick.labelsize": 12,
            "legend.fontsize": 12, "legend.title_fontsize": 12,
            "axes.titleweight": "bold"},
    )
    study = grade_boxplot(
        samples["Study time"], "studytime", STUDY_CODES, STUDY_LABELS,
        "Final grade by weekly study time", "Reported study time per week"
    )
    save_figure(study, folder / "01-study-time.png")
    save_figure(absence_plot(samples["Absences"]), folder / "02-absences.png")
    failures = grade_boxplot(
        samples["Prior failures"], "failure_group", FAILURE_GROUPS,
        FAILURE_GROUPS, "Final grade by prior class failures",
        "Prior class failures"
    )
    save_figure(failures, folder / "03-prior-failures.png")


def main() -> None:
    """Read the dataset, then print and save all EDA results."""
    matplotlib.use("Agg")
    raw = load_data(DATA_PATH)
    data = prepare_data(raw)
    samples = question_samples(data)
    tests, followups = primary_tests(samples)
    sample_sizes = pd.Series(
        {question: len(sample) for question, sample in samples.items()},
        name="rows_used",
    ).to_frame()
    sample_sizes["rows_excluded"] = len(raw) - sample_sizes["rows_used"]
    tables = {
        "data-audit": audit_data(raw, data),
        "missing-by-column": raw.isna().sum().rename("missing_cells"),
        "sample-sizes": sample_sizes,
        "quantitative-summary": quantitative_summary(data),
        "category-counts": category_counts(data),
        "study-grade-summary": grade_summary(
            samples["Study time"], "studytime"
        ),
        "failure-grade-summary": grade_summary(
            samples["Prior failures"], "failure_group"
        ),
        "primary-tests": tests,
        "study-adjacent-tests": followups,
        "sensitivity": sensitivity_summary(data),
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, table in tables.items():
        heading = name.replace("-", " ").title()
        print(f"\n{heading}")
        if table.empty:
            print("No follow-up tests were triggered.")
        else:
            formatted = table.to_string(float_format=lambda x: f"{x:.6g}")
            print(formatted)
        table.to_csv(OUTPUT_DIR / f"{name}.csv")
    save_plots(samples, OUTPUT_DIR)
    print(f"\nTables and plots saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
