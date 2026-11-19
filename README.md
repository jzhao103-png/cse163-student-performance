# Study Time Absences and Prior Failures in Portuguese Grades

This CSE 163 project explores how reported weekly study time, school absences,
and a history of class failures are associated with the final Portuguese grade
in the UCI Student Performance dataset.

## Run in VS Code

Open the repository folder and select a Python 3.12 or 3.13 interpreter. Run
these commands from the VS Code terminal.

```text
python -m pip install -r requirements.txt
python eda.py
python test_eda.py
python -m flake8 eda.py inference.py test_eda.py
```

The analysis prints its results and creates `outputs/` with three PNG plots
and CSV tables for the data audit, descriptive statistics, tests, and
sensitivity checks. The confidence intervals use 5,000 bootstrap resamples
and a fixed seed. The testing script uses hand-checked examples and checks
the bundled dataset.

## Files

| File | Purpose |
| --- | --- |
| `eda.py` | Read and validate data, print tables, and save figures |
| `inference.py` | Run the statistical tests and estimate effects and intervals |
| `test_eda.py` | Test the input, transformations, and analysis helpers |
| `data/student-por.csv` | Unmodified data for the Portuguese course |
| `data/student.txt` | UCI data dictionary for both subject files |

## Data source

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning
Repository. The data is licensed under CC BY 4.0.
[Dataset and DOI](https://doi.org/10.24432/C5TG7T).
[License](https://creativecommons.org/licenses/by/4.0/).

To retrieve a fresh copy, download the UCI archive, extract `student.zip`
from it, and put `student-por.csv` and `student.txt` from the inner
archive in `data/`.
[Download the UCI archive](https://archive.ics.uci.edu/static/public/320/student+performance.zip).

Pandas and seaborn follow the CSE 163 course examples. The inference module
uses [SciPy statistics](https://docs.scipy.org/doc/scipy/reference/stats.html).
ChatGPT and Claude helped brainstorm the project and validate the work
plan.
