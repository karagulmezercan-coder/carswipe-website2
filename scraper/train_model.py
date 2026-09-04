"""
CarSwipe — XGBoost + LightGBM Fiyat Tahmin Modeli v6
Değişiklikler:
  - Per-source dedup (cross-dataset dedup kaldırıldı) → +43K satır
  - kaynak_yil özelliği → Türkiye enflasyonu yakalanır
  - Target encoding: marka ve model_tam (smoothed)
  - Direkt yakit_tipi + kasa_tipi feature (model adından inference yerine)
  - LightGBM karşılaştırması
  - n_estimators=5000, early_stopping=100
"""

import pandas as pd
import numpy as np
import re, os, joblib, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

def load_csv(path):
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  ✓ {os.path.basename(path)}: {len(df):,} satır")
    return df

# ── Ayrıştırıcılar ───────────────────────────────────────────
def parse_boya_degisen(s):
    s = str(s).lower()
    deg = re.search(r'(\d+)\s*değişen', s)
    boy = re.search(r'(\d+)\s*boyalı', s)
    return (int(deg.group(1)) if deg else 0,
            int(boy.group(1)) if boy else 0)

def parse_motor_gucu(s):
    s = str(s).replace("HP","").strip()
    nums = re.findall(r'\d+', s)
    if not nums: return np.nan
    return np.mean([int(x) for x in nums])

def parse_motor_hacmi_sah(s):
    nums = re.findall(r'\d+', str(s))
    if not nums: return np.nan
    return np.mean([int(x) for x in nums]) / 1000

def parse_km_sah(s):
    if isinstance(s, (int, float)): return float(s)
    s = str(s).replace("km","").replace(".","").replace(",",".").strip()
    try: return float(s)
    except: return np.nan

def parse_fiyat_sah(s):
    if isinstance(s, (int, float)): return float(s)
    s = str(s).replace("TL","").replace(".","").replace(",",".").strip()
    try: return float(s)
    except: return np.nan

def parse_sehir_sah(s):
    parts = str(s).split(",")
    return parts[-1].strip() if len(parts) > 1 else str(s).strip()

def yakit_from_str(s):
    ml = str(s).lower()
    if any(x in ml for x in ["elektrik","electric"]): return 4
    if any(x in ml for x in ["hibrit","hybrid"]): return 3
    if "lpg" in ml: return 2
    if any(x in ml for x in ["dizel","diesel","tdi","cdi","hdi","dci","cdti",
                               "jtd","multijet","d4d","crdi","jtdm","bluehdi"]): return 1
    return 0

print("=" * 60)
print("CarSwipe XGBoost+LGB Model v6")
print("=" * 60)
print("\n[1] Veri yükleniyor...")

arabam     = load_csv("arabam_ilanlar.csv")
sahibinden = load_csv("sahibinden_ilanlar.csv")

common_cols = ["marka","model_tam","yil","km","fiyat","renk","sehir","satici_tipi"]

# ── Her kaynak ayrı ayrı dedup edilir ────────────────────────
def dedup_source(df, key_cols=None):
    if key_cols is None:
        key_cols = ["km","yil","fiyat","marka"]
    return df.drop_duplicates(subset=key_cols)

rows = []

if len(arabam):
    tmp = arabam[common_cols].copy()
    tmp["kaynak"] = "arabam_base"
    tmp["motor_gucu"] = 0; tmp["vites_oto"] = 0
    tmp["degisen"] = 0; tmp["boyali"] = 0; tmp["tramer"] = 0
    tmp["kasa_tipi"] = "Bilinmiyor"; tmp["yakit_tipi_str"] = "Bilinmiyor"
    tmp["motor_hacmi_parsed"] = np.nan
    tmp = dedup_source(tmp)
    rows.append(tmp)

if len(sahibinden):
    tmp = sahibinden[common_cols].copy()
    tmp["kaynak"] = "sahibinden_base"
    tmp["motor_gucu"] = 0; tmp["vites_oto"] = 0
    tmp["degisen"] = 0; tmp["boyali"] = 0; tmp["tramer"] = 0
    tmp["kasa_tipi"] = "Bilinmiyor"; tmp["yakit_tipi_str"] = "Bilinmiyor"
    tmp["motor_hacmi_parsed"] = np.nan
    tmp = dedup_source(tmp)
    rows.append(tmp)

arabam_k = load_csv("arabam_kaggle.csv")
if len(arabam_k):
    k = arabam_k.copy()
    k["model_tam"]   = k["seri"].fillna("") + " " + k["model"].fillna("")
    k["km"]          = k["kilometre"]
    k["fiyat"]       = k["fiyat"]
    k["satici_tipi"] = k["kimden"].fillna("Bilinmiyor")
    k["sehir"]       = "Bilinmiyor"
    k["renk"]        = k["renk"].fillna("Bilinmiyor")
    k["kaynak"]      = "arabam_kaggle"
    k["degisen"]     = pd.to_numeric(k["degisen_sayisi"], errors="coerce").fillna(0)
    k["boyali"]      = pd.to_numeric(k["boyali_sayisi"],  errors="coerce").fillna(0)
    k["kasa_tipi"]   = k["kasa_tipi"].fillna("Bilinmiyor")
    k["yakit_tipi_str"] = k["yakit_tipi"].fillna("Bilinmiyor")
    k["motor_gucu"]  = pd.to_numeric(k["motor_gucu"], errors="coerce").fillna(0)
    k["motor_hacmi_parsed"] = pd.to_numeric(k["motor_hacmi"], errors="coerce").fillna(np.nan)
    k["vites_oto"]   = (k["vites_tipi"].fillna("Manuel").str.lower()
                        .isin(["otomatik","yarı otomatik","cvt"])).astype(int)
    k["tramer"]      = 0
    tmp = k[common_cols + ["kaynak","degisen","boyali","kasa_tipi","yakit_tipi_str",
                           "motor_gucu","vites_oto","tramer","motor_hacmi_parsed"]].copy()
    tmp = dedup_source(tmp)
    rows.append(tmp)

sah_k = load_csv("sahibinden_kaggle.csv")
if len(sah_k):
    s = sah_k.copy()
    bd = s["boya_degisen"].fillna("").apply(parse_boya_degisen)
    s["degisen"] = [x[0] for x in bd]
    s["boyali"]  = [x[1] for x in bd]
    s["model_tam"]   = s["seri"].fillna("") + " " + s["model"].fillna("")
    s["km"]          = s["kilometre"].apply(parse_km_sah)
    s["fiyat"]       = s["fiyat"].apply(parse_fiyat_sah)
    s["sehir"]       = s["konum"].apply(parse_sehir_sah)
    s["renk"]        = s["renk"].fillna("Bilinmiyor")
    s["satici_tipi"] = s["kimden"].fillna("Bilinmiyor")
    s["kaynak"]      = "sahibinden_kaggle"
    s["kasa_tipi"]   = s["kasa_tipi"].fillna("Bilinmiyor")
    s["yakit_tipi_str"] = s["yakit_tipi"].fillna("Bilinmiyor")
    s["motor_gucu"]  = s["motor_gucu"].apply(parse_motor_gucu)
    s["motor_hacmi_parsed"] = s["motor_hacmi"].apply(parse_motor_hacmi_sah)
    s["vites_oto"]   = (s["vites_tipi"].fillna("Manuel").str.lower()
                        .isin(["otomatik","yarı otomatik","cvt"])).astype(int)
    s["tramer"]      = pd.to_numeric(s["tramer"], errors="coerce").fillna(0)
    tmp = s[["marka","model_tam","yil","km","fiyat","renk","sehir","satici_tipi",
             "kaynak","degisen","boyali","kasa_tipi","yakit_tipi_str",
             "motor_gucu","vites_oto","tramer","motor_hacmi_parsed"]].copy()
    tmp = dedup_source(tmp)
    rows.append(tmp)

sah_26 = load_csv("sahibinden_april2026.csv")
if len(sah_26):
    s = sah_26.copy()
    s["model_tam"]   = s["seri"].fillna("") + " " + s["model"].fillna("")
    s["km"]          = s["kilometre"]
    s["sehir"]       = s["konum"].fillna("Bilinmiyor").str.split(",").str[-1].str.strip()
    s["satici_tipi"] = "Bilinmiyor"
    s["renk"]        = "Bilinmiyor"
    s["kaynak"]      = "sahibinden_2026"
    s["kasa_tipi"]   = s["kasa_tipi"].replace("-","Bilinmiyor").fillna("Bilinmiyor")
    s["yakit_tipi_str"] = s["yakit_tipi"].fillna("Bilinmiyor")
    s["motor_hacmi_parsed"] = s["motor_hacmi"].fillna(0) / 1000
    s["motor_gucu"]  = pd.to_numeric(s["motor_gucu"], errors="coerce").fillna(0)
    s["vites_oto"]   = (s["vites_tipi"].fillna("Manuel").str.lower()
                        .isin(["otomatik","yarı otomatik","cvt"])).astype(int)
    s["tramer"]      = pd.to_numeric(s["tramer"], errors="coerce").fillna(0)
    s["degisen"]     = pd.to_numeric(s["degisen"], errors="coerce").fillna(0)
    s["boyali"]      = pd.to_numeric(s["boyali"],  errors="coerce").fillna(0)
    tmp = s[["marka","model_tam","yil","km","fiyat","renk","sehir","satici_tipi",
             "kaynak","degisen","boyali","kasa_tipi","yakit_tipi_str",
             "motor_gucu","vites_oto","tramer","motor_hacmi_parsed"]].copy()
    tmp = dedup_source(tmp)
    rows.append(tmp)

df = pd.concat(rows, ignore_index=True)
print(f"  Toplam (per-source dedup): {len(df):,} ilan")

# Kaynak → yaklaşık ilan yılı (enflasyon için)
KAYNAK_YIL = {
    "arabam_base": 2024,
    "sahibinden_base": 2024,
    "arabam_kaggle": 2025,
    "sahibinden_kaggle": 2025,
    "sahibinden_2026": 2026,
}
df["kaynak_yil"] = df["kaynak"].map(KAYNAK_YIL).astype(float)

HAS_EXTRA = True

# Eksik sütunlar
for col in ["degisen","boyali","motor_gucu","vites_oto","tramer"]:
    df[col] = df[col].fillna(0)
df["kasa_tipi"]     = df["kasa_tipi"].fillna("Bilinmiyor")
df["yakit_tipi_str"]= df["yakit_tipi_str"].fillna("Bilinmiyor")

print("\n[2] Veri temizleniyor...")

def parse_fiyat(s):
    if pd.isna(s): return np.nan
    if isinstance(s, (int, float)): return float(s)
    s = str(s).replace("TL","").replace(".","").replace(",",".").strip()
    try: return float(s)
    except: return np.nan

def parse_km(s):
    if pd.isna(s): return np.nan
    if isinstance(s, (int, float)): return float(s)
    s = str(s).replace("km","").replace(".","").replace(",",".").strip()
    try: return float(s)
    except: return np.nan

df["fiyat_tl"] = df["fiyat"].apply(parse_fiyat)
df["km_sayi"]  = df["km"].apply(parse_km)
df["yil"]      = pd.to_numeric(df["yil"], errors="coerce")

df = df.dropna(subset=["fiyat_tl","km_sayi","yil"])
df = df[(df["fiyat_tl"] > 50_000)  & (df["fiyat_tl"] < 50_000_000)]
df = df[(df["km_sayi"] >= 0)       & (df["km_sayi"] < 1_500_000)]
df = df[(df["yil"] >= 1990)        & (df["yil"] <= 2026)]

keep = df.groupby("marka")["fiyat_tl"].transform(
    lambda x: (x >= x.quantile(0.01)) & (x <= x.quantile(0.99))
)
df = df[keep].reset_index(drop=True)
print(f"  Temizlendikten sonra: {len(df):,} ilan")

print("\n[3] Özellikler oluşturuluyor...")

df["yas"]         = 2026 - df["yil"]
df["log_km"]      = np.log1p(df["km_sayi"])
df["km_per_year"] = df["km_sayi"] / df["yas"].clip(lower=1)
df["log_fiyat"]   = np.log1p(df["fiyat_tl"])

# Motor hacmi
def extract_cc(model):
    m = re.search(r'\b(\d+\.\d+)\b', str(model))
    if m:
        val = float(m.group(1))
        if 0.5 <= val <= 8.0: return val
    return np.nan

def engine_type(model):
    ml = str(model).lower()
    if any(x in ml for x in ["elektrik","electric"]): return 3
    if any(x in ml for x in ["hibrit","hybrid","phev","mhev","hev"]): return 2
    if any(x in ml for x in ["tsi","tdi","cdti","cdi","hdi","jtd","dci",
                               "tfsi","bluehdi","multijet","d4d","crdi","jtdm","bluetec"]): return 1
    return 0

DONANIM_MAP = {
    "authentique":1,"ambiance":1,"access":1,"attraction":1,
    "joy":2,"easy":2,"expression":2,"essential":2,"pop":2,"life":2,"live":2,"ls":2,
    "comfort":3,"touch":3,"trend":3,"techline":3,"core":3,"style":3,"feel":3,
    "zen":3,"motion":3,"pure":3,"plus":3,"edition":3,"connect":3,"icon":3,
    "optimal":3,"urban":3,"stepway":3,"techno":3,"shine":3,"drive":3,"match":3,
    "elegance":4,"titanium":4,"highline":4,"prestige":4,"ambition":4,"intens":4,
    "signature":4,"sportline":4,"gt line":4,"gtline":4,"s line":4,"premium":4,
    "dream":4,"comfortline":4,"executive":4,"excellence":4,"advanced":4,
    "evolution":4,"fr":4,"gt":4,"sport":4,"dynamic":4,"trendline":4,
    "lounge":5,"r-line":5,"individual":5,"exclusive":5,"initiale":5,"ultimate":5,
    "luxury":5,"black":5,"platinum":5,"amg":5,"m sport":5,"m-sport":5,
    "nismo":5,"gti":5,"full":5,"esprit alpine":5,"alpine":5,"limited":5,
}
def donanim_skoru(model):
    ml = str(model).lower()
    best = 0
    for k, v in DONANIM_MAP.items():
        if k in ml: best = max(best, v)
    return best if best > 0 else 3

df["motor_hacmi_model"] = df["model_tam"].apply(extract_cc)
df["motor_hacmi"] = df["motor_hacmi_model"].combine_first(df["motor_hacmi_parsed"])
df["motor_hacmi"] = df["motor_hacmi"].fillna(df["motor_hacmi"].median())

df["motor_tipi"] = df["model_tam"].apply(engine_type)
df["donanim"]    = df["model_tam"].apply(donanim_skoru)

# Yakıt tipi: direkt veri varsa kullan, yoksa model adından çıkar
df["yakit_from_data"] = df["yakit_tipi_str"].apply(yakit_from_str)
df["yakit_from_name"] = df["model_tam"].apply(lambda m: yakit_from_str(m))
df["yakit"] = df["yakit_from_data"].where(df["yakit_from_data"] > 0, df["yakit_from_name"])

# Kasa encode
KASA_MAP = {
    "sedan":0,"hatchback/3":1,"hatchback/5":2,"suv":3,"crossover":4,
    "mpv":5,"station wagon":6,"pick-up":7,"cabriolet":8,"coupe":9,
    "roadster":10,"minivan":11,"panelvan":12
}

def infer_kasa(row):
    kt = str(row.get("kasa_tipi","")).lower().strip()
    if kt and kt not in ("bilinmiyor","nan","","-"):
        v = KASA_MAP.get(kt)
        if v is not None: return v
    ml = str(row.get("model_tam","")).lower()
    if any(x in ml for x in ["hatchback","hb"]): return 2
    if "sedan" in ml: return 0
    if any(x in ml for x in [" suv","4x4","awd","4wd"]): return 3
    if "crossover" in ml: return 4
    if any(x in ml for x in ["station","sw","touring","avant","variant","estate","combi"]): return 6
    if any(x in ml for x in ["coupe","coupé","coup"]): return 9
    if any(x in ml for x in ["cabrio","cabriolet","roadster","spider"]): return 8
    return np.nan

df["kasa_enc"] = df.apply(infer_kasa, axis=1).astype(float)

for col in ["marka","renk","sehir","satici_tipi","model_tam"]:
    df[col] = df[col].fillna("Bilinmiyor").astype(str).str.strip()

# Kaza/boya türetme
df["degisen"]    = df["degisen"].clip(0, 20)
df["boyali"]     = df["boyali"].clip(0, 20)
df["kaza_var"]   = (df["degisen"] > 0).astype(int)
df["boyali_var"] = (df["boyali"] > 2).astype(int)
df["log_tramer"] = np.log1p(df["tramer"])

print("\n[4] Encoding + Target Encoding...")

# Label encoding
cat_cols = ["marka","renk","sehir","satici_tipi"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

# Model frekansı
freq = df["model_tam"].value_counts()
df["model_freq"] = df["model_tam"].map(freq)

# Target encoding (smoothed, 5-fold cross-val)
def target_encode_cv(df, col, target="log_fiyat", n_folds=5, smoothing=10):
    global_mean = df[target].mean()
    result = pd.Series(global_mean, index=df.index)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(df):
        tr = df.iloc[tr_idx]
        stats = tr.groupby(col)[target].agg(["mean","count"])
        # Bayesian smoothing
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        result.iloc[val_idx] = df.iloc[val_idx][col].map(smooth).fillna(global_mean)
    return result

df["marka_target"]    = target_encode_cv(df, "marka")
df["model_target"]    = target_encode_cv(df, "model_tam")
df["marka_cc"]        = df["marka_enc"] * df["motor_hacmi"]

kasa_known = df["kasa_enc"].notna().sum()
print(f"  Kasa bilineni: {kasa_known:,} / {len(df):,}")

print("\n[5] Model eğitiliyor...")

FEATURES = [
    # Temel araç yaşı/km
    "yas", "log_km", "km_per_year",
    # Motor
    "motor_hacmi", "motor_tipi", "motor_gucu",
    # Donanım/yakıt
    "donanim", "yakit",
    # Frekans ve target encoding
    "model_freq", "marka_target", "model_target",
    # Label encoding
    "marka_enc", "renk_enc", "sehir_enc", "satici_tipi_enc",
    # Kaza/boya
    "degisen", "boyali", "kaza_var", "boyali_var",
    # Kasa + vites
    "kasa_enc", "vites_oto",
    # Tramer + pazar zamanı
    "log_tramer", "kaynak_yil",
    # Interaction
    "marka_cc",
]  # 24 özellik

X = df[FEATURES]
y = df["log_fiyat"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"  Eğitim: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── XGBoost ────────────────────────────────────────────────────
print("\n  [XGBoost eğitiliyor...]")
xgb_model = xgb.XGBRegressor(
    n_estimators          = 5000,
    max_depth             = 7,
    learning_rate         = 0.02,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 5,
    reg_alpha             = 0.05,
    reg_lambda            = 1.0,
    random_state          = 42,
    n_jobs                = -1,
    verbosity             = 0,
    early_stopping_rounds = 100,
    tree_method           = "hist",
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=500)
print(f"  XGB en iyi iterasyon: {xgb_model.best_iteration}")

def metrics(model, X_test, y_test):
    log_pred = model.predict(X_test)
    y_pred   = np.expm1(log_pred)
    y_true   = np.expm1(y_test)
    mae    = mean_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)
    mape   = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    medape = np.median(np.abs((y_true - y_pred) / y_true)) * 100
    errors = np.abs((y_true - y_pred) / y_true) * 100
    return dict(mae=mae, r2=r2, mape=mape, medape=medape,
                p10=(errors<10).mean()*100,
                p20=(errors<20).mean()*100,
                p30=(errors<30).mean()*100)

xgb_m = metrics(xgb_model, X_test, y_test)
print(f"\n  XGB → MAPE: {xgb_m['mape']:.1f}% | MedAPE: {xgb_m['medape']:.1f}% | R²: {xgb_m['r2']:.4f}")
print(f"        <10%: %{xgb_m['p10']:.0f} | <20%: %{xgb_m['p20']:.0f} | <30%: %{xgb_m['p30']:.0f}")

best_model = xgb_model
best_mape  = xgb_m["mape"]
model_type = "xgb"

# ── LightGBM ───────────────────────────────────────────────────
if HAS_LGB:
    print("\n  [LightGBM eğitiliyor...]")
    lgb_model = lgb.LGBMRegressor(
        n_estimators      = 5000,
        max_depth         = 8,
        learning_rate     = 0.02,
        num_leaves        = 127,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_samples = 20,
        reg_alpha         = 0.05,
        reg_lambda        = 1.0,
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = -1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )
    lgb_m = metrics(lgb_model, X_test, y_test)
    print(f"  LGB → MAPE: {lgb_m['mape']:.1f}% | MedAPE: {lgb_m['medape']:.1f}% | R²: {lgb_m['r2']:.4f}")
    print(f"        <10%: %{lgb_m['p10']:.0f} | <20%: %{lgb_m['p20']:.0f} | <30%: %{lgb_m['p30']:.0f}")

    if lgb_m["mape"] < best_mape:
        best_model = lgb_model
        best_mape  = lgb_m["mape"]
        model_type = "lgb"
        print("  → LightGBM kazandı!")
    else:
        print("  → XGBoost kazandı!")

print("\n[6] Özellik Önemi (kazanan model)...")
if model_type == "xgb":
    fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
else:
    fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in fi.head(12).items():
    print(f"    {feat:28s} {imp:.4f}")

print("\n[7] Model kaydediliyor...")
joblib.dump(best_model, "carswipe_model.pkl")
joblib.dump(encoders,   "carswipe_encoders.pkl")
joblib.dump(FEATURES,   "carswipe_features.pkl")
joblib.dump(sorted(df["marka"].unique().tolist()), "carswipe_markalar.pkl")
joblib.dump(model_type, "carswipe_model_type.pkl")
joblib.dump(True,       "carswipe_has_extra.pkl")
joblib.dump(KASA_MAP,   "carswipe_kasa_map.pkl")
# Target encoding istatistiklerini kaydet (inference için)
marka_stats = df.groupby("marka")["log_fiyat"].agg(["mean","count"])
model_stats  = df.groupby("model_tam")["log_fiyat"].agg(["mean","count"])
global_mean  = df["log_fiyat"].mean()
joblib.dump((marka_stats, model_stats, global_mean, 10), "carswipe_target_enc.pkl")
print("  ✓ Tüm dosyalar kaydedildi.")

print("\n" + "=" * 60)
kaynaklar = df["kaynak"].value_counts().to_dict()
for k, v in kaynaklar.items():
    print(f"  {k}: {v:,} ilan")
winner = xgb_m if model_type == "xgb" else lgb_m
print(f"\n✓ Tamamlandı! {len(df):,} ilan | Kazanan: {model_type.upper()}")
print(f"  MAPE: {winner['mape']:.1f}% | MedAPE: {winner['medape']:.1f}% | R²: {winner['r2']:.4f}")
print("=" * 60)
