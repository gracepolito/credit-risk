
#!/usr/bin/env python3
import argparse
import pandas as pd
from credit_model.fico_bucketer import fit_fico_buckets_mse, save_boundaries

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fico_col", default="fico_score")
    ap.add_argument("--default_col", default="default")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boundaries_json", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    res = fit_fico_buckets_mse(df, args.fico_col, args.default_col, args.K)

    # Report
    print("\n=== FICO Bucketer (MSE / 1-D k-means) ===")
    print("Boundaries (right-closed, ascending FICO):")
    for i, b in enumerate(res.boundaries, start=1):
        print(f"  Bucket {i}: <= {b if pd.notnull(b) else '+inf'}")
    print("\nCounts & PD per rating (1=best):")
    for r in range(1, args.K+1):
        print(f"  Rating {r}: n={res.counts[r-1]}, PD≈{res.pd_per_rating[r-1]:.4f}")

    # Save
    df_out = df.copy()
    df_out["rating"] = res.ratings
    df_out.to_csv(args.out, index=False)
    save_boundaries(res.boundaries, args.boundaries_json)
    print(f"\nSaved: {args.out}")
    print(f"Saved boundaries: {args.boundaries_json}")

if __name__ == "__main__":
    main()
