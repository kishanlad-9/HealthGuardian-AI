# datasets/

Raw and processed datasets used to train the ML models. Not committed to git
(see `.gitignore`) — regenerated via download scripts, not versioned, to
keep the repo lean and avoid redistributing UCI data directly.

## Heart Disease (Milestone 4)

```bash
python ml/download_heart_dataset.py
```

Downloads `heart_disease_uci.csv` — the UCI Cleveland Heart Disease dataset
(303 patients, 13 clinical features + binary target), from a community
mirror on GitHub (the original UCI archive isn't always reachable from
every environment, so a GitHub-hosted copy is used for reproducibility).

**Known data-quality caveat**: in this specific mirror, the values `ca=4`
and `thal=0` are widely reported in public discussion of this dataset to
actually represent missing-value codes from the original UCI encoding,
not real clinical categories. `ml/preprocessing.py` treats them as their
own one-hot category rather than silently imputing them.

**Critical fix applied (Milestone 5)**: this mirror's raw `target` column
is inverted (`0`=disease-present, `1`=disease-absent) - corrected in
`ml/preprocessing.load_heart_data()`. See `saved_models/README.md` and
the main `README.md`'s Milestone 5 section for the full story.

## Diabetes (Milestone 7)

```bash
python ml/download_diabetes_dataset.py
```

Downloads `pima_diabetes.csv` — the UCI Pima Indians Diabetes dataset (768
women of Pima Indian heritage, age 21+, 8 features + binary Outcome), from
Jason Brownlee's well-known ML datasets mirror on GitHub. Target encoding
is the intuitive convention here (`1`=diabetes-positive) - verified against
two synthetic patient profiles before trusting it, given the heart disease
dataset's inversion surprise.

**Known data-quality issue, handled (not just noted)**: `glucose`,
`blood_pressure`, `skin_thickness`, `insulin`, and `bmi` use `0` as a
missing-value placeholder — a living patient cannot have 0 blood pressure.
Verified empirically (see `ml/preprocessing_diabetes.py`): insulin is
missing in 48.7% of rows, skin_thickness in 29.6%. `pregnancies=0` is a
legitimate value and is NOT touched by this fix. Handled via
`SimpleImputer(strategy="median")` inside the training pipeline (fit on
the training fold only, to avoid leaking test-set statistics).
