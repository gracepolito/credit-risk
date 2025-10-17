
#!/usr/bin/env python3
import argparse
import pandas as pd
from credit_model.pd_model import train_logistic_pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="default")
    ap.add_argument("--drop", nargs="*", default=[])
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    res = train_logistic_pd(df, target=args.target, drop_cols=args.drop)

    print("\n=== Logistic PD Model ===")
    print("Features:", res.features)
    print("ROC-AUC:", round(res.roc_auc, 4))
    print("\nClassification Report:\n", res.report)

if __name__ == "__main__":
    main()
