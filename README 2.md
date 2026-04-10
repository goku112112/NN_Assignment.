# Bank Marketing Prediction 🏦

I built this project to predict whether a customer will subscribe to a term deposit, using the UCI Bank Marketing dataset (`bank-full.csv`).

## What I was trying to do

The dataset comes from a bank's phone marketing campaign. My goal was to figure out who is actually likely to say **yes** to a term deposit — so that future campaigns can focus on the right people instead of cold-calling everyone.

## My Approach

The first thing I noticed was the class imbalance — about 88% of customers said "no" and only ~12% said "yes". I kept that in mind throughout the whole process.

I also made a deliberate decision to drop the `duration` column early on. It tells you how long the call lasted, but you only know that *after* the call ends — so it's useless for making predictions beforehand. Keeping it would've inflated my results artificially.

**What I did to clean the data:**
- Replaced `-1` in `pdays` with `NaN` — it actually means the customer was never previously contacted
- Replaced all `"unknown"` entries with `NaN`, then filled them using mode imputation
- Clipped outliers in `balance`, `campaign`, and `previous`
- One-hot encoded all categorical features

## Models I Tried

**Logistic Regression** — I started with this as a simple baseline.

**Random Forest (200 trees)** — This became my main model. It handles mixed feature types well and gives feature importances without any extra work.

## What I Found

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~89% | — |
| Random Forest | ~91% | ~0.93 |

Random Forest came out on top. A ROC-AUC of ~0.93 tells me it's doing a solid job at ranking likely subscribers — not just getting the easy "no" cases right.

**The features that mattered most:**
- `poutcome` (previous campaign outcome) — strongest signal by a wide margin
- `balance` — how much money they have in their account
- `age` and `job` type
- `contact` method and `month` of the call

The biggest takeaway: if someone said yes in a past campaign, they're very likely to say yes again. Obvious in hindsight, but good to see the data confirm it.

## Files

- `logistic_model.pkl` — saved logistic regression model
- Main notebook has the full pipeline from raw CSV to evaluation

## A few notes

- I loaded the dataset from Google Drive (extracted from a zip file)
- I originally imported PyTorch with plans to build a neural net, but the sklearn models performed well enough that I didn't end up needing it
- Training a neural net would be a logical next step if I wanted to push performance further
