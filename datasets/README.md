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
own one-hot category rather than silently imputing them — documented
there and here rather than "fixed" quietly, since imputation choices
should be visible, not hidden inside a preprocessing step.

## Diabetes (Milestone 7 — not yet added)

- `pima_diabetes.csv` — UCI Pima Indians Diabetes dataset
