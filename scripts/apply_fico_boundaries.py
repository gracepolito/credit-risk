
#!/usr/bin/env python3
import argparse
import pandas as pd
from credit_model.fico_bucketer import load_boundaries, apply_boundaries

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fico_col", default="fico_score")
    ap.add_argument("--boundaries_json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    boundaries = load_boundaries(args.boundaries_json)
    df["rating"] = apply_boundaries(df[args.fico_col], boundaries)
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
