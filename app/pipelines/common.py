# df is a pandas DataFrame
from datetime import datetime
from pathlib import Path


def slice_feature(df, feat):
    """
    Select columns/features from dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
    feat : str or list[str]

    Returns
    -------
    pandas.DataFrame
    """

    # Single column
    if isinstance(feat, str):
        return df[[feat]]

    # Multiple columns
    return df[feat]

def log_status(job_path, message):
    log_file = Path(job_path) / "job_status.log"

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")