import filecmp
import glob
import pathlib as p
import sys

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


# ----------------------------------------------------------------------
# tolerant comparator --------------------------------------------------
# ----------------------------------------------------------------------
def equal_csv(left, right, *, atol=1e-9, rtol=0, sort_cols=None, verbose=True):
    """
    Compare two CSVs with numeric tolerance and report row-wise diffs if mismatched.
    Rows are sorted by `sort_cols` (or all columns by default).
    Returns True if equal, False if different.
    """
    df_l = pd.read_csv(left)
    df_r = pd.read_csv(right)

    # Ensure same columns in same order
    if set(df_l.columns) != set(df_r.columns):
        print(f"❌ Columns mismatch: {left}")
        return False
    df_l = df_l[df_r.columns]  # reorder to match

    # Sort rows
    sort_cols = sort_cols or df_l.columns.tolist()
    df_l_sorted = df_l.sort_values(by=sort_cols).reset_index(drop=True)
    df_r_sorted = df_r.sort_values(by=sort_cols).reset_index(drop=True)

    try:
        assert_frame_equal(
            df_l_sorted,
            df_r_sorted,
            check_dtype=False,
            atol=atol,
            rtol=rtol,
            obj=f"CSV[{left}]",
        )
        return True
    except AssertionError:
        if verbose:
            print(f"⚠️  Differences in {left}")
            # Show row-by-row diffs
            diff_rows = []
            for i, (row_l, row_r) in enumerate(
                zip(df_l_sorted.values, df_r_sorted.values)
            ):
                if not all(
                    pd.isna(row_l[j])
                    and pd.isna(row_r[j])
                    or isinstance(row_l[j], str)
                    and row_l[j] == row_r[j]
                    or isinstance(row_l[j], (int, float))
                    and isinstance(row_r[j], (int, float))
                    and abs(row_l[j] - row_r[j]) <= atol + rtol * abs(row_r[j])
                    for j in range(len(row_l))
                ):
                    diff_rows.append(i)

            for i in diff_rows[:10]:  # Show first 10 row diffs only
                print(f"  ❗ Row {i}:")
                for col in df_l_sorted.columns:
                    val_l = df_l_sorted.at[i, col]
                    val_r = df_r_sorted.at[i, col]
                    if isinstance(val_l, float) and isinstance(val_r, float):
                        if (
                            not pd.isna(val_l)
                            and not pd.isna(val_r)
                            and not np.isclose(val_l, val_r, atol=atol, rtol=rtol)
                        ):
                            print(f"    {col}: {val_l:.6g} vs {val_r:.6g}")
                    elif val_l != val_r:
                        print(f"    {col}: {val_l} vs {val_r}")
            if len(diff_rows) > 10:
                print(f"    ... {len(diff_rows) - 10} more differing rows")
        return False


def main(prod_dir="input_fuel_pathway_data/production", base_dir="baseline_production"):
    diff = False
    for f in glob.glob(f"{prod_dir}/*.csv"):
        g = p.Path(base_dir, p.Path(f).name)
        if not equal_csv(f, g):
            print("⚠️ changed:", f)
            diff = True
    if diff:
        sys.exit("mismatch")  # non-zero exit for CI / bash
    print("✓ all outputs identical")


if __name__ == "__main__":
    main()
