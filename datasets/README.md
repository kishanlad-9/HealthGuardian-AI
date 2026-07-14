# datasets/

Raw and processed datasets used to train the ML models. Not committed to git
(see `.gitignore`) — data files are regenerated/downloaded, not versioned,
to keep the repo lean and avoid redistributing UCI data directly.

Expected contents once Milestone 4 (Heart Disease model) begins:
- `heart_disease_uci.csv` — UCI Heart Disease dataset
- `pima_diabetes.csv` — UCI Pima Indians Diabetes dataset (Milestone 7)

A `download_datasets.py` script will be added in Milestone 4 to fetch these
reproducibly rather than requiring manual download.
