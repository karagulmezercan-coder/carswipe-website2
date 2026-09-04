"""
CarSwipe Fiyat Tahmin API — FastAPI
Kullanım:
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000

Endpoint:
    POST /predict
    GET  /predict?marka=Renault&model_tam=Clio+1.0+TCe+Touch&yil=2020&km=85000
    GET  /health
    GET  /markalalar
"""
import re, os, math, logging
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("carswipe")

# ---------------------------------------------------------------------------
# Model yükleme
# ---------------------------------------------------------------------------
SCRAPER_DIR = os.path.dirname(os.path.abspath(__file__))

def load(name):
    path = os.path.join(SCRAPER_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact bulunamadı: {path}")
    return joblib.load(path)

_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Model yükleniyor...")
    _state["model"]       = load("carswipe_model.pkl")
    _state["encoders"]    = load("carswipe_encoders.pkl")
    _state["features"]    = load("carswipe_features.pkl")
    _state["model_type"]  = load("carswipe_model_type.pkl")
    _state["has_extra"]   = load("carswipe_has_extra.pkl")
    _state["kasa_map"]    = load("carswipe_kasa_map.pkl")
    _state["markalar"]    = load("carswipe_markalar.pkl")
    ms, mt, gm, s = load("carswipe_target_enc.pkl")
    smooth_marka  = (ms["count"]*ms["mean"] + s*gm) / (ms["count"] + s)
    smooth_model  = (mt["count"]*mt["mean"] + s*gm) / (mt["count"] + s)
    _state["marka_stats"] = smooth_marka
    _state["model_stats"] = smooth_model
    _state["global_mean"] = gm
    try:
        _state["model_freq"] = load("carswipe_model_freq.pkl")
    except Exception:
        _state["model_freq"] = {}
    _load_trinkoto_cache()
    log.info(f"Model hazır — tür: {_state['model_type']}, {len(_state['features'])} feature")
    yield
    _state.clear()

app = FastAPI(
    title="CarSwipe Araç Fiyat API",
    version="1.0",
    description="LightGBM tabanlı Türk ikinci el araç fiyat tahmini",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar (train_fast.py ile birebir aynı)
# ---------------------------------------------------------------------------
DONANIM_MAP = {
    "authentique":1,"ambiance":1,"access":1,"joy":2,"easy":2,"essential":2,"pop":2,
    "comfort":3,"touch":3,"trend":3,"techline":3,"core":3,"style":3,"feel":3,"plus":3,
    "elegance":4,"titanium":4,"highline":4,"prestige":4,"ambition":4,"signature":4,
    "sport":4,"dynamic":4,"gt":4,"fr":4,"premium":4,"s line":4,"sportline":4,
    "lounge":5,"r-line":5,"exclusive":5,"luxury":5,"amg":5,"m sport":5,"gti":5,"full":5,
}

def donanim_enc(model_str: str) -> int:
    ml = model_str.lower()
    return max((v for k, v in DONANIM_MAP.items() if k in ml), default=3)

def extract_cc(model_str: str) -> float:
    r = re.search(r'\b(\d+\.\d+)\b', model_str)
    if r:
        v = float(r.group(1))
        if 0.5 <= v <= 8.0:
            return v
    return float("nan")

def engine_type(model_str: str) -> int:
    ml = model_str.lower()
    if any(x in ml for x in ["elektrik", "electric"]): return 3
    if any(x in ml for x in ["hibrit", "hybrid"]):     return 2
    if any(x in ml for x in ["tsi","tdi","cdti","cdi","hdi","dci","crdi"]): return 1
    return 0

def yakit_enc(s: str) -> int:
    ml = s.lower()
    if any(x in ml for x in ["elektrik","electric"]): return 4
    if any(x in ml for x in ["hibrit","hybrid"]):     return 3
    if "lpg" in ml:                                    return 2
    if any(x in ml for x in ["dizel","diesel","tdi","cdi","hdi","dci"]): return 1
    return 0

def kasa_enc_fn(kasa_tipi: str, model_str: str, kasa_map: dict) -> float:
    kt = kasa_tipi.lower().strip()
    v = kasa_map.get(kt)
    if v is not None:
        return float(v)
    ml = model_str.lower()
    if any(x in ml for x in ["hatchback","hb"]): return 2.0
    if "sedan" in ml:                             return 0.0
    if any(x in ml for x in [" suv","4x4"]):     return 3.0
    if "crossover" in ml:                         return 4.0
    if any(x in ml for x in ["station","touring","avant","variant"]): return 6.0
    if any(x in ml for x in ["coupe","coupé"]):   return 9.0
    return float("nan")

def safe_le(le, val: str):
    """LabelEncoder'da görülmemiş etiket → en yakın or 0."""
    val = str(val).strip()
    classes = list(le.classes_)
    if val in classes:
        return le.transform([val])[0]
    # yoksa 0
    return 0

# ---------------------------------------------------------------------------
# Request / Response şemaları
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    marka:       str  = Field(..., example="Renault")
    model_tam:   str  = Field(..., example="Clio 1.0 TCe Touch")
    yil:         int  = Field(..., ge=1990, le=2030, example=2020)
    km:          float= Field(..., ge=0, le=1_500_000, example=85_000)
    # Opsiyonel alanlar
    renk:        str  = Field("Bilinmiyor", example="Beyaz")
    sehir:       str  = Field("Bilinmiyor", example="İstanbul")
    satici_tipi: str  = Field("Bilinmiyor", example="Galeriden")
    kasa_tipi:   str  = Field("Bilinmiyor", example="Hatchback/5")
    yakit_tipi:  str  = Field("Bilinmiyor", example="Benzin")
    vites_tipi:  str  = Field("Manuel", example="Manuel")   # Manuel / Otomatik
    motor_gucu:  float= Field(0.0,  ge=0, le=2000, example=100)
    motor_hacmi: float= Field(0.0,  ge=0, le=10,   example=1.0)
    vites_oto:   int  = Field(0,    ge=0, le=1,     example=0)
    degisen:     int  = Field(0,    ge=0, le=20,    example=0)
    boyali:      int  = Field(0,    ge=0, le=20,    example=0)
    tramer:      float= Field(0.0,  ge=0,            example=0)

class PredictResponse(BaseModel):
    tahmin_tl:   float
    alt_bant_tl: float
    ust_bant_tl: float
    mape_beklenti: str = "~8.7%"
    input_ozeti: dict

# ---------------------------------------------------------------------------
# Tahmin yardımcısı
# ---------------------------------------------------------------------------
def build_features(req: PredictRequest) -> "pd.DataFrame":
    s = _state
    # vites_oto: vites_tipi string'inden türet (yoksa direkt vites_oto kullan)
    if hasattr(req, 'vites_tipi') and req.vites_tipi:
        req.vites_oto = 1 if req.vites_tipi.lower() in ['otomatik', 'yarı otomatik', 'cvt'] else 0
    yas = 2025 - req.yil
    log_km = math.log1p(req.km)
    km_per_year = req.km / max(yas, 1)

    # motor_hacmi: önce model_tam'dan çıkar, yoksa input
    mh_model = extract_cc(req.model_tam)
    if math.isnan(mh_model):
        motor_hacmi = req.motor_hacmi if req.motor_hacmi > 0 else 1.6  # default median
    else:
        motor_hacmi = mh_model

    motor_tipi = engine_type(req.model_tam)
    donanim    = donanim_enc(req.model_tam)

    yakit_d = yakit_enc(req.yakit_tipi)
    yakit_m = yakit_enc(req.model_tam)
    yakit   = yakit_d if yakit_d > 0 else yakit_m

    # Encoding
    marka_enc      = safe_le(s["encoders"]["marka"],       req.marka)
    renk_enc       = safe_le(s["encoders"]["renk"],        req.renk)
    sehir_enc      = safe_le(s["encoders"]["sehir"],       req.sehir)
    satici_tipi_enc= safe_le(s["encoders"]["satici_tipi"], req.satici_tipi)

    # Frequency (model_tam) — gerçek eğitim frekansı, bilinmiyorsa medyan (5)
    model_freq = int(s.get("model_freq", {}).get(req.model_tam, 5))

    # Target encoding
    gm = s["global_mean"]
    marka_target = float(s["marka_stats"].get(req.marka,  gm))
    model_target = float(s["model_stats"].get(req.model_tam, gm))

    marka_cc = marka_enc * motor_hacmi

    kasa = kasa_enc_fn(req.kasa_tipi, req.model_tam, s["kasa_map"])

    degisen    = float(min(req.degisen, 20))
    boyali     = float(min(req.boyali,  20))
    kaza_var   = int(degisen > 0)
    boyali_var = int(boyali  > 2)
    log_tramer = math.log1p(req.tramer)

    # Extra features
    motor_gucu = req.motor_gucu
    power_per_liter = min(motor_gucu / max(motor_hacmi, 0.1), 500)
    yas_sq      = yas ** 2
    log_km_yas  = log_km * yas

    feat_map = {
        "yas":             yas,
        "log_km":          log_km,
        "km_per_year":     km_per_year,
        "motor_hacmi":     motor_hacmi,
        "motor_tipi":      motor_tipi,
        "motor_gucu":      motor_gucu,
        "donanim":         donanim,
        "yakit":           yakit,
        "model_freq":      model_freq,
        "marka_target":    marka_target,
        "model_target":    model_target,
        "marka_enc":       marka_enc,
        "renk_enc":        renk_enc,
        "sehir_enc":       sehir_enc,
        "satici_tipi_enc": satici_tipi_enc,
        "degisen":         degisen,
        "boyali":          boyali,
        "kaza_var":        kaza_var,
        "boyali_var":      boyali_var,
        "kasa_enc":        kasa,
        "vites_oto":       req.vites_oto,
        "log_tramer":      log_tramer,
        "marka_cc":        marka_cc,
        "power_per_liter": power_per_liter,
        "yas_sq":          yas_sq,
        "log_km_yas":      log_km_yas,
    }
    return pd.DataFrame([feat_map], columns=s["features"])

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_type": _state.get("model_type"), "features": len(_state.get("features", []))}

@app.get("/markalar")
def markalar():
    return {"markalar": _state.get("markalar", [])}

@app.post("/predict", response_model=PredictResponse)
def predict_post(req: PredictRequest):
    return _predict(req)

@app.get("/predict", response_model=PredictResponse)
def predict_get(
    marka:       str   = Query(...),
    model_tam:   str   = Query(...),
    yil:         int   = Query(...),
    km:          float = Query(...),
    renk:        str   = Query("Bilinmiyor"),
    sehir:       str   = Query("Bilinmiyor"),
    satici_tipi: str   = Query("Bilinmiyor"),
    kasa_tipi:   str   = Query("Bilinmiyor"),
    yakit_tipi:  str   = Query("Bilinmiyor"),
    motor_gucu:  float = Query(0.0),
    motor_hacmi: float = Query(0.0),
    vites_oto:   int   = Query(0),
    degisen:     int   = Query(0),
    boyali:      int   = Query(0),
    tramer:      float = Query(0.0),
    vites_tipi:  str   = Query("Manuel"),
):
    req = PredictRequest(
        marka=marka, model_tam=model_tam, yil=yil, km=km,
        renk=renk, sehir=sehir, satici_tipi=satici_tipi,
        kasa_tipi=kasa_tipi, yakit_tipi=yakit_tipi,
        motor_gucu=motor_gucu, motor_hacmi=motor_hacmi,
        vites_oto=vites_oto, degisen=degisen, boyali=boyali,
        tramer=tramer, vites_tipi=vites_tipi,
    )
    return _predict(req)

def _predict(req: PredictRequest) -> PredictResponse:
    try:
        X = build_features(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature oluşturulamadı: {e}")

    try:
        log_pred = float(_state["model"].predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {e}")

    tahmin = math.expm1(log_pred)
    # ±8.7% MAPE'ye dayalı bant (1 MAPE aralığı)
    band = 0.087
    alt  = tahmin * (1 - band)
    ust  = tahmin * (1 + band)

    return PredictResponse(
        tahmin_tl   = round(tahmin),
        alt_bant_tl = round(alt),
        ust_bant_tl = round(ust),
        input_ozeti = {
            "marka":     req.marka,
            "model_tam": req.model_tam,
            "yil":       req.yil,
            "km":        req.km,
        },
    )

# ---------------------------------------------------------------------------
# Trinkoto piyasa doğrulama — offline cache dosyasından
# ---------------------------------------------------------------------------
import json as _json

TRINKOTO_CACHE_PATH = os.path.join(SCRAPER_DIR, "trinkoto_cache.json")
_trinkoto_cache: dict = {}

def _load_trinkoto_cache():
    global _trinkoto_cache
    if os.path.exists(TRINKOTO_CACHE_PATH):
        try:
            _trinkoto_cache = _json.loads(open(TRINKOTO_CACHE_PATH, encoding="utf-8").read())
            log.info(f"Trinkoto cache yüklendi: {len(_trinkoto_cache)} kayıt")
        except Exception as e:
            log.warning(f"Trinkoto cache yüklenemedi: {e}")
    else:
        log.info("Trinkoto cache bulunamadı — /validate trinkoto bandı olmadan çalışacak")
        log.info("  Cache oluşturmak için: python trinkoto_scraper.py --test")

def _trinkoto_slug(s: str) -> str:
    return re.sub(r'\s+', '-', s.strip().lower())

def _lookup_trinkoto(marka: str, model_kelime: str) -> Optional[dict]:
    """Cache'den marka/model fiyat bandı döndür."""
    key = f"{_trinkoto_slug(marka)}/{_trinkoto_slug(model_kelime)}"
    data = _trinkoto_cache.get(key)
    if not data:
        return None
    return {
        "trinkoto_min": data.get("min"),
        "trinkoto_max": data.get("max"),
        "trinkoto_ort": data.get("ort"),
        "fiyat_sayisi": data.get("n", 0),
        "versiyonlar":  data.get("versiyonlar", [])[:5],
    }

@app.get("/validate")
def validate(
    marka:     str   = Query(..., example="Renault"),
    model_tam: str   = Query(..., example="Clio 1.0 TCe Touch"),
    yil:       int   = Query(..., example=2020),
    km:        float = Query(..., example=85_000),
):
    """Model tahmini + Trinkoto piyasa bandını karşılaştır."""
    req = PredictRequest(marka=marka, model_tam=model_tam, yil=yil, km=km)
    pred = _predict(req)

    model_kelime = model_tam.split()[0] if model_tam else ""
    trinkoto = _lookup_trinkoto(marka, model_kelime)

    result = {
        "model_tahmini": {
            "tahmin_tl":   pred.tahmin_tl,
            "alt_bant_tl": pred.alt_bant_tl,
            "ust_bant_tl": pred.ust_bant_tl,
        },
        "trinkoto_piyasa": trinkoto,
        "karsilastirma": None,
    }

    if trinkoto and trinkoto.get("trinkoto_ort"):
        fark_pct = (pred.tahmin_tl - trinkoto["trinkoto_ort"]) / trinkoto["trinkoto_ort"] * 100
        result["karsilastirma"] = {
            "model_vs_trinkoto_ort_pct": round(fark_pct, 1),
            "yorum": (
                "Model trinkoto ortalamasına yakın ✓" if abs(fark_pct) < 15
                else f"Model trinkoto ortalamasından %{abs(fark_pct):.0f} {'yüksek' if fark_pct>0 else 'düşük'} ⚠"
            ),
        }
    return result

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
