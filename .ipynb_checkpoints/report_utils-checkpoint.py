import pandas as pd


def summarize_feature(df: pd.DataFrame, feature: str) -> dict:
    """
    Compute basic summary statistics for a single feature.

    For numeric features, returns count, unique, missing, mean, min, and max.
    For non-numeric features, returns count, unique, and missing.
    """
    series = df[feature]
    summary = {
        "feature": feature,
        "dtype": str(series.dtype),
        "count": int(series.count()),
        "unique": int(series.nunique(dropna=True)),
        "missing": int(series.isna().sum()),
    }

    # Add numeric stats when applicable
    if pd.api.types.is_numeric_dtype(series):
        summary.update(
            {
                "mean": float(series.mean()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    else:
        summary.update(
            {
                "mean": None,
                "min": None,
                "max": None,
            }
        )

    return summary


def compare_slices_mean(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    feature: str,
    name1: str = "slice_1",
    name2: str = "slice_2",
) -> dict:
    """
    Compare the mean of a numeric feature between two slices.

    Returns a dictionary with the mean in each slice and the difference.
    """
    s1 = df1[feature]
    s2 = df2[feature]

    mean1 = float(s1.mean())
    mean2 = float(s2.mean())
    diff = mean1 - mean2

    return {
        "feature": feature,
        "slice_1_name": name1,
        "slice_2_name": name2,
        "slice_1_mean": mean1,
        "slice_2_mean": mean2,
        "difference": diff,
    }
