import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logger import logging
import config

PAPER_FEATURES = [
    "Avg min between sent tnx",
    "Avg min between received tnx",
    "Time Diff between first and last (Mins)",
    "Sent tnx",
    "Received Tnx",
    "Number of Created Contracts",
    "Total ERC20 tnxs",
    "min value received",
    "max value received",
    "avg val received",
    "min val sent",
    "max val sent",
    "avg val sent",
    "ERC20 min val sent",
    "ERC20 max val sent",
    "ERC20 avg val sent",
    "Unique Received From Addresses",
    "Unique Sent To Addresses",
    "FLAG",
]

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    logging.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath)

    logging.info(f"Original shape: {df.shape}")

    df.columns = df.columns.str.strip()

    cols_to_drop = ["Index", "Address", "Unnamed: 0"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    nan_count = int(df.isna().sum().sum())
    if nan_count > 0:
        logging.info(f"Filling {nan_count} NaNs with 0")
        df = df.fillna(0)

    missing = [c for c in PAPER_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required paper columns: {missing}")

    df = df[PAPER_FEATURES].copy()

    # force numeric for safety
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.isna().sum().sum() > 0:
        logging.info("NaNs created during numeric coercion; filling with 0")
        df = df.fillna(0)

    # keep label binary/int
    df["FLAG"] = df["FLAG"].astype(int)

    zero_var_cols = [c for c in df.columns if c != "FLAG" and df[c].nunique() <= 1]
    if zero_var_cols:
        logging.info(f"Dropping zero-variance feature columns: {zero_var_cols}")
        df = df.drop(columns=zero_var_cols)

    logging.info(f"Final cleaned shape: {df.shape}")
    logging.info(f"Final feature columns: {[c for c in df.columns if c != 'FLAG']}")

    return df


def get_train_test_split(
    df: pd.DataFrame,
    seed: int,
):
    """
    75 / 25 split with stratification (paper requirement)
    """
    logging.info(f"Train/test split with seed={seed}")

    if "FLAG" not in df.columns:
        raise ValueError("FLAG column missing")

    X = df.drop(columns=["FLAG"])
    logging.info(f"Feature columns ({len(X.columns)}): {list(X.columns)}")
    y = df["FLAG"]

    with open("feature_columns.txt", "w", encoding="utf-8") as f:
        for col in X.columns:
            f.write(f"{col}\n")

    logging.info(f"Overall fraud rate: {df['FLAG'].mean():.6f}")

    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=config.DATA_SPLIT_RATIO,random_state=seed,stratify=y,)

    logging.info(f"Train size={len(X_train)} fraud_rate={y_train.mean():.6f}")

    logging.info(f"Test size={len(X_test)} fraud_rate={y_test.mean():.6f}")


    # # -------------------------
    # # Feature normalization
    # # -------------------------

    # scaler = StandardScaler()

    # X_train_scaled = pd.DataFrame(
    #     scaler.fit_transform(X_train),
    #     columns=X_train.columns,
    #     index=X_train.index,
    # )

    # X_test_scaled = pd.DataFrame(
    #     scaler.transform(X_test),
    #     columns=X_test.columns,
    #     index=X_test.index,
    # )

    # logging.info("Feature normalization applied (StandardScaler)")

    return X_train, X_test, y_train, y_test
