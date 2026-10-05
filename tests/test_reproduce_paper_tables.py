"""Recompute the example segment tables and compare their fingerprints with tests/expected/."""
import os, sys, json, hashlib
import pandas as pd, cv2
try:
    import pytest
    parametrize = pytest.mark.parametrize
except ImportError:  # so the file also runs as a plain script
    def parametrize(_name, _values):
        return lambda f: f

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from rootsupport import mask_to_segments, trace_paths, support_counts

CASES = ["2025-04-07", "2025-04-21"]


def fingerprint(df: pd.DataFrame) -> dict:
    """Same definition that produced tests/expected/*.json from the paper tables."""
    d = df.sort_values("seg_id")
    def f(v):
        return "" if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v))
    def t(v):
        return "" if pd.isna(v) else str(v)
    rows = ["|".join([str(int(r.seg_id)), f(r.x_1), f(r.y_1), t(r.type_1), f(r.x_2), f(r.y_2), t(r.type_2),
                      f(r.geodesic_len), f(r.euclidean_len), str(int(0 if pd.isna(r.support_count) else r.support_count))])
            for _, r in d.iterrows()]
    s = d.support_count.fillna(0).astype(int)
    return {"n_segments": int(len(d)),
            "support_histogram": {str(k): int(v) for k, v in s.value_counts().sort_index().items()},
            "n_terminal": int(((d.type_1 == "TIP") ^ (d.type_2 == "TIP")).sum()),
            "n_isolated": int(((d.type_1 == "TIP") & (d.type_2 == "TIP")).sum()),
            "sha256_of_table": hashlib.sha256("\n".join(rows).encode()).hexdigest()}


def run_case(date):
    mask = cv2.imread(os.path.join(ROOT, "examples", "data", f"aruco_109_date_{date}_mask.png"), cv2.IMREAD_GRAYSCALE)
    skeleton, segs, seg_info, df = mask_to_segments(mask)
    _, seg_to_paths, _ = trace_paths(seg_info)
    df["support_count"] = df["seg_id"].map(support_counts(seg_info, seg_to_paths)).astype(int)
    exp = json.load(open(os.path.join(ROOT, "tests", "expected", f"aruco_109_date_{date}_fingerprint.json")))
    return df, exp


@parametrize("date", CASES)
def test_segments_and_support_match_paper(date):
    df, exp = run_case(date)
    got = fingerprint(df)
    assert got["n_segments"] == exp["n_segments"], f"{got['n_segments']} segments recomputed vs {exp['n_segments']} in the paper table"
    assert got["support_histogram"] == exp["support_histogram"], "support histogram differs from the paper table"
    assert got["sha256_of_table"] == exp["sha256_of_table"], "segment ends, lengths or support differ from the paper table"


def test_support_invariants():
    df, _ = run_case(CASES[0])
    term = (df.type_1 == "TIP") ^ (df.type_2 == "TIP")
    iso = (df.type_1 == "TIP") & (df.type_2 == "TIP")
    assert (df.loc[term, "support_count"] == 1).all(), "terminal segments must have support 1"
    assert (df.loc[iso, "support_count"] == 0).all(), "isolated segments must have support 0"


if __name__ == "__main__":
    for d in CASES:
        test_segments_and_support_match_paper(d)
        print(f"{d}: segments, ends, lengths and support match the paper's table")
    test_support_invariants()
    print("invariants hold (terminal = 1, isolated = 0)")
