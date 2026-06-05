from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import datetime
import os

# Inisialisasi Aplikasi API
app = FastAPI(title="HDB Price Prediction API")

# Setup CORS agar Landing Page (HTML) dari luar diizinkan menarik data
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Dalam tahap produksi, ganti "*" dengan URL landing page Anda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Struktur Data yang diharapkan dari Landing Page
class PropertyData(BaseModel):
    town: str
    flat_type: str
    area: float
    storey: int
    lease: int

# Load Model saat API menyala
model_path = "hdb_model.pkl"
try:
    with open(model_path, "rb") as f:
        assets = pickle.load(f)
        model = assets["model"]
        encoder = assets["encoder"]
        print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model: {e}")
    model, encoder = None, None

@app.get("/")
def read_root():
    return {"status": "API berjalan lancar!", "message": "Gunakan endpoint /predict dengan method POST."}

@app.post("/predict")
def predict_price(data: PropertyData):
    if not model or not encoder:
        return {"error": "Model Machine Learning belum siap di server."}

    try:
        # Siapkan DataFrame sesuai format yang dipelajari model saat training
        current_year = datetime.datetime.now().year
        current_month = datetime.datetime.now().month

        input_df = pd.DataFrame([{
            'town': data.town,
            'flat_type': data.flat_type,
            'floor_area_sqm': data.area,
            'mid_storey': data.storey,
            'lease_years': data.lease,
            'year': current_year,
            'month_num': current_month
        }])

        # Encode teks kategori menjadi angka
        input_df[['town', 'flat_type']] = encoder.transform(input_df[['town', 'flat_type']])

        # Eksekusi Prediksi
        predicted_price = model.predict(input_df)[0]

        # Kirim balik hasil prediksi ke Landing Page
        return {"predicted_price": round(predicted_price, 2)}

    except Exception as e:
        return {"error": str(e)}