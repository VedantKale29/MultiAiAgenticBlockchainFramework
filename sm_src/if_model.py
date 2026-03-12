import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from  logger import logging
import  config as config

class IFModel:
    def __init__(self, seed: int, y_train: pd.Series):
        self.seed = seed
        
        # Paper requirement: contamination set to observed fraud rate in training set
        self.contamination = max(1e-6, y_train.mean())
        
        # Paper requirement: max_samples = min(2048, #normals)
        num_normals = (y_train == 0).sum()
        self.max_samples = min(config.IF_MAX_SAMPLES, num_normals)
        logging.info(f"Normal samples={num_normals}")
        
        self.model = IsolationForest(
            n_estimators=config.IF_N_ESTIMATORS,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=-1
        )
        # Scaler for mapping anomaly scores to [0, 1]
        self.scaler = MinMaxScaler()
        
        logging.info(f"Initialized Isolation Forest (trees={config.IF_N_ESTIMATORS}, max_samples={self.max_samples}, contamination={self.contamination:.6f}, seed={self.seed})")

    def train(self, X_train, y_train):

        logging.info("Training Isolation Forest baseline...")

        X_normals = X_train[y_train == 0]

        logging.info(f"Normal samples={len(X_normals)}")

        self.model.fit(X_normals)

        raw_scores = -self.model.score_samples(X_train)

        self.scaler.fit(raw_scores.reshape(-1, 1))

        logging.info("Isolation Forest training complete.")

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns the scaled anomaly score in the range [0, 1].
        """
        raw_scores = -self.model.score_samples(X)
        scaled_scores = self.scaler.transform(raw_scores.reshape(-1, 1)).flatten()
        
        # Clip to strictly enforce [0, 1] bounds in case of extreme test outliers
        return np.clip(scaled_scores, 0, 1)