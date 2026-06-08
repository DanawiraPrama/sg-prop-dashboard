from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

# 1. Inisialisasi API
app = FastAPI()

# Buka gerbang CORS biar website loe bisa narik data
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load Model & Encoder Versi Baru (Data 10 Tahun)
with open("hdb_model.pkl", "rb") as f:
    assets = pickle.load(f)
    model = assets["model"]
    le_town = assets["le_town"]
    le_flat = assets["le_flat"]

# 3. Format Data yang dikirim dari HTML
class PredictRequest(BaseModel):
    town: str
    flat_type: str
    area: float
    storey: int
    lease: int

# 4. Endpoint untuk Prediksi
@app.post("/predict")
def predict_price(req: PredictRequest):
    try:
        # Ubah teks (Town & Flat Type) jadi angka pakai Encoder dari model
        t_code = le_town.transform([req.town])[0]
        f_code = le_flat.transform([req.flat_type])[0]
        
        # Susun data sesuai urutan training: 
        # ['town_code', 'flat_code', 'floor_area_sqm', 'remaining_lease', 'storey_num']
        features = np.array([[t_code, f_code, req.area, req.lease, req.storey]])
        
        # Prediksi harga
        prediction = model.predict(features)[0]
        
        return {"predicted_price": float(prediction)}
    
    except Exception as e:
        # Kalau ada error, kembalikan response error (biar ga jadi NaN di frontend)
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))