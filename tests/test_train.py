import os
import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.train import train
from src.serve import app

FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    """
    Tao dataset nho voi cung schema Wine Quality de su dung trong test.

    pytest cung cap `tmp_path` la mot thu muc tam thoi, tu dong xoa sau khi test ket thuc.
    Ham nay dung du lieu ngau nhien nen khong can ket noi GCS hay tai file CSV thuc.
    """
    rng = np.random.default_rng(0)
    n = 200

    # 1. Tao mang X (n, 12)
    X = rng.random((n, len(FEATURE_NAMES)))

    # 2. Tao mang y gom n phan tu [0, 1, 2]
    y = rng.integers(0, 3, size=n)

    # 3. Tao DataFrame voi cot target
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    # 4. Luu tap train (160) va eval (40)
    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    return train_path, eval_path


def test_train_returns_float(tmp_path):
    """Kiem tra ham train() tra ve mot so thuc nam trong [0.0, 1.0]."""
    train_path, eval_path = _make_temp_data(tmp_path)
    acc = train(
        {"n_estimators": 10, "max_depth": 3, "model_type": "random_forest"},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    """Kiem tra file outputs/metrics.json duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3, "model_type": "random_forest"},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "class_distribution" in metrics


def test_model_file_created(tmp_path):
    """Kiem tra file models/model.pkl duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3, "model_type": "random_forest"},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert os.path.exists("models/model.pkl")


def test_multi_model_support(tmp_path):
    """Bonus 2: Kiem tra huan luyen voi cac thuat toan khac (HistGradientBoosting, LogisticRegression)."""
    train_path, eval_path = _make_temp_data(tmp_path)
    
    # HistGradientBoosting
    acc_gb = train(
        {"max_iter": 20, "max_depth": 3, "model_type": "hist_gradient_boosting"},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(acc_gb, float)
    assert 0.0 <= acc_gb <= 1.0

    # LogisticRegression
    acc_lr = train(
        {"max_iter": 100, "model_type": "logistic_regression"},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(acc_lr, float)
    assert 0.0 <= acc_lr <= 1.0


def test_serve_api():
    """Kiem tra API FastAPI /health va /predict."""
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok"}

    # 2. Predict voi 12 dac trung hop le
    sample_features = [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0.0]
    res_predict = client.post("/predict", json={"features": sample_features})
    assert res_predict.status_code == 200
    data = res_predict.json()
    assert "prediction" in data
    assert "label" in data
    assert data["prediction"] in [0, 1, 2]
    assert data["label"] in ["thap", "trung_binh", "cao"]

    # 3. Predict voi so luong dac trung sai (11 thay vi 12)
    bad_features = [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
    res_bad = client.post("/predict", json={"features": bad_features})
    assert res_bad.status_code == 400
