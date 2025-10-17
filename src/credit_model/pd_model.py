
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

@dataclass
class PDModelResult:
    model: LogisticRegression
    roc_auc: float
    report: str
    features: List[str]

def train_logistic_pd(df: pd.DataFrame, target: str, drop_cols: Optional[List[str]] = None,
                      test_size: float = 0.2, random_state: int = 42) -> PDModelResult:
    drop_cols = drop_cols or []
    X = df.drop(columns=[target] + drop_cols)
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)
    return PDModelResult(model, auc, report, X.columns.tolist())

def expected_loss(model: LogisticRegression, loan_features: pd.DataFrame, recovery_rate: float = 0.10) -> pd.DataFrame:
    out = loan_features.copy()
    pd_hat = model.predict_proba(out)[:, 1]
    out["PD"] = pd_hat
    out["EAD"] = out["loan_amt_outstanding"]
    out["LGD"] = 1 - recovery_rate
    out["Expected_Loss"] = out["PD"] * out["EAD"] * out["LGD"]
    return out[["PD", "EAD", "LGD", "Expected_Loss"]]
