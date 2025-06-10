import sys, pandas as pd, filecmp, glob, pathlib as p
import pandas as pd
from pandas.testing import assert_frame_equal

# ----------------------------------------------------------------------
# tolerant comparator --------------------------------------------------
# ----------------------------------------------------------------------
def equal_csv(left, right, *, atol=1e-9, rtol=0):
    """
    Return True when the two CSV files are identical *within* absolute
    tolerance `atol` (default 1 × 10⁻9) and relative tolerance `rtol`.
    Non-numeric columns must match exactly.
    """
    df_l = pd.read_csv(left).sort_index(axis=1)
    df_r = pd.read_csv(right).sort_index(axis=1)
    try:
        assert_frame_equal(
            df_l,
            df_r,
            check_dtype=False,    # ignore dtype differences like int64 ↔ float64
            atol=atol,
            rtol=rtol,
            obj=f"CSV[{left}]",
        )
        return True
    except AssertionError:
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

