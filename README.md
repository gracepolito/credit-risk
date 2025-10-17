
# Credit Risk PD & FICO Bucketing (Prototype)

A clean, reproducible prototype for **personal-loan default prediction** and **mortgage FICO bucketing**:

- **PD model** (Logistic Regression baseline) on tabular borrower features
- **Expected Loss** function: `EL = PD × EAD × (1 − Recovery)` (defaults to 10% recovery)
- **FICO Bucketizer** (MSE / 1-D k-means): maps `fico_score` → categorical **ratings** (1 = best)
- Lightweight scripts + notebook, intended as a starting point for model validation + productionization

> ⚠️ Data is kept private. See `data/README.md` for how to place your CSV locally.
