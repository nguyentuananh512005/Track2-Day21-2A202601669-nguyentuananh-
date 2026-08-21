import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

EVAL_THRESHOLD = 0.70


def check_data_drift_and_distribution(y_train: pd.Series, y_eval: pd.Series = None) -> dict:
    """
    Kiem tra ty le phan phoi nhan (Bonus 5 - Data Drift / Imbalance warning).
    1. In canh bao ro rang neu bat ky lop nao chiem < 10% tong mau trong tap train.
    2. Neu co y_eval, so sanh su chech lech ty le giua train va eval (phat hien Data Drift neu chenh > 15%).
    """
    total_train = len(y_train)
    dist_train = y_train.value_counts(normalize=True).to_dict()
    warnings_list = []
    
    print("[DATA AUDIT] Phan phoi nhan tap huan luyen:")
    for cls in [0, 1, 2]:
        ratio_tr = dist_train.get(cls, 0.0)
        print(f"  - Lop {cls}: {ratio_tr * 100:.2f}% ({int(ratio_tr * total_train)} mau)")
        if ratio_tr < 0.10:
            msg = f"[CANH BAO CLASS IMBALANCE] Lop {cls} chiem {ratio_tr * 100:.2f}% (< 10% nguong an toan)!"
            print(f"    -> {msg}")
            warnings_list.append(msg)

    # Kiem tra Data Drift neu co tap eval
    dist_eval = {}
    drift_detected = False
    if y_eval is not None:
        dist_eval = y_eval.value_counts(normalize=True).to_dict()
        print("[DATA AUDIT] So sanh phan phoi Train vs Eval (Data Drift check):")
        for cls in [0, 1, 2]:
            r_tr = dist_train.get(cls, 0.0)
            r_ev = dist_eval.get(cls, 0.0)
            diff = abs(r_tr - r_ev)
            print(f"  - Lop {cls}: Train={r_tr*100:.2f}% vs Eval={r_ev*100:.2f}% (Chenh lech: {diff*100:.2f}%)")
            if diff > 0.15:
                drift_msg = f"[CANH BAO DATA DRIFT] Lop {cls} chenh lech {diff*100:.2f}% (> 15% nguong canh bao drift)!"
                print(f"    -> {drift_msg}")
                warnings_list.append(drift_msg)
                drift_detected = True

    return {
        "distribution_train": dist_train,
        "distribution_eval": dist_eval,
        "drift_detected": drift_detected,
        "warnings": warnings_list,
    }


check_class_distribution = check_data_drift_and_distribution


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so va loai mo hinh.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """
    # 1. Doc du lieu
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2. Tach dac trung va nhan
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: Kiem tra phan phoi lop & canh bao lech lac du lieu (Data Drift audit)
    audit_results = check_data_drift_and_distribution(y_train, y_eval)

    with mlflow.start_run():
        # 3. Ghi nhan sieu tham so
        mlflow.log_params(params)

        # 4. Khoi tao mo hinh (Bonus 2: Ho tro nhieu thuat toan)
        model_type = params.get("model_type", "random_forest")
        
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 5),
                min_samples_split=params.get("min_samples_split", 2),
                random_state=42,
            )
        elif model_type in ["hist_gradient_boosting", "gradient_boosting"]:
            model = HistGradientBoostingClassifier(
                max_iter=params.get("max_iter", 100),
                max_depth=params.get("max_depth", 5),
                random_state=42,
            )
        elif model_type == "logistic_regression":
            model = LogisticRegression(
                max_iter=params.get("max_iter", 500),
                random_state=42,
            )
        else:
            raise ValueError(f"Khong ho tro model_type: {model_type}")

        # Huan luyen mo hinh
        model.fit(X_train, y_train)

        # 5. Du doan va danh gia
        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        # Bonus 3: Chi tiet bao cao phan loai
        report_str = classification_report(y_eval, preds, digits=4, zero_division=0)
        cm = confusion_matrix(y_eval, preds).tolist()

        # 6. Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        for cls, ratio in audit_results["distribution_train"].items():
            mlflow.log_metric(f"class_{cls}_ratio", float(ratio))

        # Log artifact model
        mlflow.sklearn.log_model(model, "model")

        # 7. In ket qua ra man hinh
        print(f"[{model_type.upper()}] Accuracy: {acc:.4f} | F1: {f1:.4f}")
        if acc >= EVAL_THRESHOLD:
            print(f"-> DAT NGUONG EVALUATION GATE ({acc:.4f} >= {EVAL_THRESHOLD})")
        else:
            print(f"-> CHUA DAT NGUONG EVALUATION GATE ({acc:.4f} < {EVAL_THRESHOLD})")

        # 8. Luu metrics ra file outputs/metrics.json
        os.makedirs("outputs", exist_ok=True)
        metrics_payload = {
            "accuracy": acc,
            "f1_score": f1,
            "eval_threshold": EVAL_THRESHOLD,
            "model_type": model_type,
            "class_distribution": audit_results["distribution_train"],
            "class_distribution_eval": audit_results["distribution_eval"],
            "data_drift_detected": audit_results["drift_detected"],
            "confusion_matrix": cm,
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics_payload, f, indent=2)

        # Bonus 3: Luu text report
        with open("outputs/report.txt", "w") as f:
            f.write(f"=== MLOPS MODEL EVALUATION REPORT ===\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Accuracy  : {acc:.4f}\n")
            f.write(f"F1-Score  : {f1:.4f}\n\n")
            f.write("=== CLASSIFICATION REPORT ===\n")
            f.write(report_str + "\n\n")
            f.write(f"=== CONFUSION MATRIX ===\n{cm}\n")

        # 9. Luu mo hinh ra file models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
