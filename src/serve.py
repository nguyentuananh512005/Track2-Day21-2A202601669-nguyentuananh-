from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(
    title="Wine Quality MLOps Inference API",
    description="FastAPI REST service phục vụ dự đoán phân loại chất lượng rượu vang",
    version="1.0.0",
)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/model.pkl"))
LOCAL_FALLBACK_MODEL = "models/model.pkl"


def download_model():
    """
    Tải file model.pkl từ GCS về máy khi server khởi động.
    Hỗ trợ fallback sang mô hình cục bộ nếu chạy trong môi trường kiểm thử/cục bộ.
    """
    if GCS_BUCKET:
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(GCS_MODEL_KEY)
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            blob.download_to_filename(MODEL_PATH)
            print(f"[GCS] Model da duoc tai xuong tu gs://{GCS_BUCKET}/{GCS_MODEL_KEY} -> {MODEL_PATH}")
            return
        except Exception as e:
            print(f"[GCS WARNING] Khong the tai tu GCS: {e}. Dang chuyen sang kiem tra local fallback...")

    # Local fallback
    if os.path.exists(MODEL_PATH):
        print(f"[LOCAL] Su dung model tai {MODEL_PATH}")
    elif os.path.exists(LOCAL_FALLBACK_MODEL):
        print(f"[LOCAL] Su dung model fallback tai {LOCAL_FALLBACK_MODEL}")
    else:
        print("[WARNING] Chua tim thay file model. Model se duoc load khi co file.")


def get_model():
    try:
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        elif os.path.exists(LOCAL_FALLBACK_MODEL):
            return joblib.load(LOCAL_FALLBACK_MODEL)
    except Exception as e:
        print(f"[WARNING] Error loading model: {e}")
    return None


download_model()
model = get_model()


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    global model
    if model is None:
        model = get_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model chua duoc san sang de suy luan.")

    # 1. Kiem tra so luong dac trung
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 12 features (wine quality), but got {len(req.features)}."
        )

    # 2. Du doan
    try:
        pred = int(model.predict([req.features])[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi du doan: {str(e)}")

    # 3. Anh xa nhan
    label_mapping = {
        0: "thap",
        1: "trung_binh",
        2: "cao",
    }
    
    return {
        "prediction": pred,
        "label": label_mapping.get(pred, "khong_xac_dinh"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
