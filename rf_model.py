import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from logger import logging
import config

class RFModel:
    def __init__(self, seed: int):
        self.seed = seed
        # Setup RF exactly as specified in the paper
        self.model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            class_weight=config.RF_CLASS_WEIGHT,
            random_state=self.seed,
            n_jobs=-1  # Use all cores to speed up training
        )
        logging.info(f"Initialized Random Forest (trees={config.RF_N_ESTIMATORS}, class_weight={config.RF_CLASS_WEIGHT}, seed={self.seed})")

    def log_feature_importance(self, X_train: pd.DataFrame, output_path: str, top_k: int = 15):
        importances = self.model.feature_importances_
        feat_imp = (
            pd.DataFrame({
                "feature": X_train.columns,
                "importance": importances
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        logging.info("Top feature importances:")
        for _, row in feat_imp.head(top_k).iterrows():
            logging.info(f"{row['feature']}: {row['importance']:.6f}")

        feat_imp.to_csv(output_path, index=False)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        logging.info("Training Random Forest baseline...")
        self.model.fit(X_train, y_train)
        logging.info(f"Samples={len(X_train)}, features={X_train.shape[1]}")
        logging.info("Random Forest training complete.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns the probability of the transaction being fraudulent (Class 1).
        """
        return self.model.predict_proba(X)[:, 1]