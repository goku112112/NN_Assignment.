# Bank Marketing Prediction 

Predicting whether a customer will subscribe to a term deposit based on the UCI Bank Marketing dataset (`bank-full.csv`).



## What's the goal?

The bank ran a phone marketing campaign and we want to predict who actually says **yes** to a term deposit — so future campaigns can be more targeted.



## Approach

The dataset is pretty imbalanced (~88% "no", ~12% "yes"), so we kept that in mind throughout. We also dropped the `duration` column since it leaks information — you only know call duration *after* the call ends, which makes it useless for real predictions.

**Preprocessing steps:**
- Replaced `-1` in `pdays` with `NaN` (it actually means "not contacted")
- Replaced `"unknown"` values with `NaN`, then imputed using mode
- Clipped outliers in `balance`, `campaign`, and `previous`
- One-hot encoded all categorical features

## Models Used

**Logistic Regression** — simple baseline, trained with imputed data.

**Random Forest (200 trees)** — the main model. Handles mixed feature types well and gives feature importances out of the box.

## Findings

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~89% | — |
| Random Forest | ~91% | ~0.93 |

Random Forest clearly wins, and the ROC-AUC of ~0.93 means it's quite good at ranking likely subscribers.

**Top features that mattered most:**
- `poutcome` (outcome of previous campaign) — by far the strongest signal
- `balance` — account balance
- `age` and `job` type
- `contact` method and `month` of call

> If someone said yes before, they're very likely to say yes again. Not surprising, but good to confirm.


## Files

- `logistic_model.pkl` — saved logistic regression model
- Main notebook contains full pipeline from raw CSV to evaluation

## Notes

- Dataset loaded from Google Drive, extracted from a zip
- PyTorch was imported but the final models used scikit-learn
- A neural net would be a natural next step if you want to push accuracy further
