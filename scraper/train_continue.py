"""
CarSwipe v7 — 2 aşamalı eğitim (LightGBM checkpoint devam)
Aşama 1: 3000 iter → model kaydedilir
Aşama 2: model'den devam → 3000 iter daha
"""
import pandas as pd, numpy as np, re, os, joblib, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

PHASE = int(os.environ.get("PHASE","1"))
print(f"=== PHASE {PHASE} ===")

def load_csv(path):
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  ✓ {os.path.basename(path)}: {len(df):,}")
    return df

def parse_boya_degisen(s):
    s = str(s).lower()
    d = re.search(r'(\d+)\s*değişen', s)
    b = re.search(r'(\d+)\s*boyalı', s)
    return (int(d.group(1)) if d else 0, int(b.group(1)) if b else 0)

def parse_motor_gucu(s):
    s = str(s).replace("HP","").strip()
    n = re.findall(r'\d+', s)
    return np.mean([int(x) for x in n]) if n else np.nan

def parse_motor_hacmi_sah(s):
    n = re.findall(r'\d+', str(s))
    return np.mean([int(x) for x in n]) / 1000 if n else np.nan

def pf(s):
    if pd.isna(s): return np.nan
    if isinstance(s,(int,float)): return float(s)
    s = str(s).replace("TL","").replace(".","").replace(",",".").strip()
    try: return float(s)
    except: return np.nan

def pk(s):
    if pd.isna(s): return np.nan
    if isinstance(s,(int,float)): return float(s)
    s = str(s).replace("km","").replace(".","").replace(",",".").strip()
    try: return float(s)
    except: return np.nan

def yakit_enc(s):
    ml = str(s).lower()
    if any(x in ml for x in ["elektrik","electric"]): return 4
    if any(x in ml for x in ["hibrit","hybrid"]): return 3
    if "lpg" in ml: return 2
    if any(x in ml for x in ["dizel","diesel","tdi","cdi","hdi","dci"]): return 1
    return 0

DONANIM_MAP = {
    "authentique":1,"ambiance":1,"access":1,"joy":2,"easy":2,"essential":2,"pop":2,
    "comfort":3,"touch":3,"trend":3,"techline":3,"core":3,"style":3,"feel":3,"plus":3,
    "elegance":4,"titanium":4,"highline":4,"prestige":4,"ambition":4,"signature":4,
    "sport":4,"dynamic":4,"gt":4,"fr":4,"premium":4,"s line":4,"sportline":4,
    "lounge":5,"r-line":5,"exclusive":5,"luxury":5,"amg":5,"m sport":5,"gti":5,"full":5,
}
def donanim(m):
    ml = str(m).lower()
    return max((v for k,v in DONANIM_MAP.items() if k in ml), default=3)

KASA_MAP = {
    "sedan":0,"hatchback/3":1,"hatchback/5":2,"suv":3,"crossover":4,
    "mpv":5,"station wagon":6,"pick-up":7,"cabriolet":8,"coupe":9,
    "roadster":10,"minivan":11,"panelvan":12,
}

CHECKPOINT = "carswipe_checkpoint.pkl"
if PHASE == 2 and os.path.exists(CHECKPOINT):
    print("[Phase 2] Checkpoint yükleniyor, veri atlanıyor...")
    model_prev, encoders, FEATURES, X_train, X_test, y_train, y_test = joblib.load(CHECKPOINT)
    booster = model_prev.booster_
    print(f"  Checkpoint: {model_prev.best_iteration_} iter tamamlandı")
    model2 = lgb.LGBMRegressor(n_estimators=1500, **dict(
        max_depth=10, num_leaves=255, learning_rate=0.025,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=25, reg_alpha=0.05, reg_lambda=0.8,
        random_state=42, n_jobs=-1, verbosity=-1,
    ))
    model2.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        init_model=booster,
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)],
    )
    model = model2
    print(f"  Toplam ağaç: {model.best_iteration_}")
    print("[6] Sonuçlar...")
    yp = np.expm1(model.predict(X_test))
    yt = np.expm1(y_test)
    mape   = np.mean(np.abs((yt-yp)/yt))*100
    medape = np.median(np.abs((yt-yp)/yt))*100
    r2     = r2_score(yt, yp)
    err    = np.abs((yt-yp)/yt)*100
    print(f"  MAPE: {mape:.1f}% | MedAPE: {medape:.1f}% | R²: {r2:.4f}")
    print(f"  <10%: %{(err<10).mean()*100:.0f} | <20%: %{(err<20).mean()*100:.0f} | <30%: %{(err<30).mean()*100:.0f}")
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n  Top 12:")
    for feat, imp in fi.head(12).items():
        print(f"    {feat:28s} {imp:.0f}")
    # Kaydediliyor (encoders + marka listesi checkpoint'ten alındı; df yok ama markalar önceki model'den)
    prev_markalar = joblib.load("carswipe_markalar.pkl") if os.path.exists("carswipe_markalar.pkl") else []
    prev_kasa     = joblib.load("carswipe_kasa_map.pkl")  if os.path.exists("carswipe_kasa_map.pkl")  else KASA_MAP
    prev_te       = joblib.load("carswipe_target_enc.pkl") if os.path.exists("carswipe_target_enc.pkl") else None
    joblib.dump(model,        "carswipe_model.pkl")
    joblib.dump(encoders,     "carswipe_encoders.pkl")
    joblib.dump(FEATURES,     "carswipe_features.pkl")
    joblib.dump(prev_markalar,"carswipe_markalar.pkl")
    joblib.dump("lgb",        "carswipe_model_type.pkl")
    joblib.dump(True,         "carswipe_has_extra.pkl")
    if prev_te: joblib.dump(prev_te, "carswipe_target_enc.pkl")
    print(f"  ✓ Final model kaydedildi! MAPE: {mape:.1f}%")
    import sys; sys.exit(0)

print("[1] Veri yükleniyor...")
arabam     = load_csv("arabam_ilanlar.csv")
sahibinden = load_csv("sahibinden_ilanlar.csv")
arabam_k   = load_csv("arabam_kaggle.csv")
sah_k      = load_csv("sahibinden_kaggle.csv")
sah_26     = load_csv("sahibinden_april2026.csv")

rows = []

def base_row(df, kaynak):
    t = df[["marka","model_tam","yil","km","fiyat","renk","sehir","satici_tipi"]].copy()
    t["kaynak"] = kaynak
    for c in ["degisen","boyali","motor_gucu","vites_oto","tramer"]:
        t[c] = 0.0
    t["kasa_tipi"] = "Bilinmiyor"
    t["yakit_tipi_str"] = "Bilinmiyor"
    t["motor_hacmi_parsed"] = np.nan
    return t.drop_duplicates(subset=["km","yil","fiyat","marka"])

if len(arabam):     rows.append(base_row(arabam,     "arabam_base"))
if len(sahibinden): rows.append(base_row(sahibinden, "sahibinden_base"))

if len(arabam_k):
    k = arabam_k.copy()
    k["model_tam"] = k["seri"].fillna("")+" "+k["model"].fillna("")
    k["km"]        = k["kilometre"]
    k["satici_tipi"] = k["kimden"].fillna("Bilinmiyor")
    k["sehir"]     = "Bilinmiyor"
    k["renk"]      = k["renk"].fillna("Bilinmiyor")
    k["kaynak"]    = "arabam_kaggle"
    k["degisen"]   = pd.to_numeric(k["degisen_sayisi"], errors="coerce").fillna(0)
    k["boyali"]    = pd.to_numeric(k["boyali_sayisi"],  errors="coerce").fillna(0)
    k["kasa_tipi"] = k["kasa_tipi"].fillna("Bilinmiyor")
    k["yakit_tipi_str"] = k["yakit_tipi"].fillna("Bilinmiyor")
    k["motor_gucu"]= pd.to_numeric(k["motor_gucu"], errors="coerce").fillna(0)
    k["motor_hacmi_parsed"] = pd.to_numeric(k["motor_hacmi"], errors="coerce").fillna(np.nan)
    k["vites_oto"] = k["vites_tipi"].fillna("Manuel").str.lower().isin(["otomatik","yarı otomatik","cvt"]).astype(int)
    k["tramer"]    = 0.0
    t = k[["marka","model_tam","yil","km","fiyat","renk","sehir","satici_tipi",
            "kaynak","degisen","boyali","kasa_tipi","yakit_tipi_str",
            "motor_gucu","vites_oto","tramer","motor_hacmi_parsed"]].copy()
    rows.append(t.drop_duplicates(subset=["km","yil","fiyat","marka"]))

if len(sah_k):
    s = sah_k.copy()
    bd = s["boya_degisen"].fillna("").apply(parse_boya_degisen)
    s["degisen"] = [x[0] for x in bd]
    s["boyali"]  = [x[1] for x in bd]
    s["model_tam"] = s["seri"].fillna("")+" "+s["model"].fillna("")
    s["km"]      = s["kilometre"].apply(pk)
    s["fiyat"]   = s["fiyat"].apply(pf)
    s["sehir"]   = s["konum"].apply(lambda x: str(x).split(",")[-1].strip())
    s["renk"]    = s["renk"].fillna("Bilinmiyor")
    s["satici_tipi"] = s["kimden"].fillna("Bilinmiyor")
    s["kaynak"]  = "sahibinden_kaggle"
    s["kasa_tipi"] = s["kasa_tipi"].fillna("Bilinmiyor")
    s["yakit_tipi_str"] = s["yakit_tipi"].fillna("Bilinmiyor")
    s["motor_gucu"] = s["motor_gucu"].apply(parse_motor_gucu)
    s["motor_hacmi_parsed"] = s["motor_hacmi"].apply(parse_motor_hacmi_sah)
    s["vites_oto"] = s["vites_tipi"].fillna("Manuel").str.lower().isin(["otomatik","yarı otomatik","cvt"]).astype(int)
    s["tramer"]  = pd.to_numeric(s["tramer"], errors="coerce").fillna(0)
    t = s[["marka","model_tam","yil","km","fiyat","renk","sehir","satici_tipi",
            "kaynak","degisen","boyali","kasa_tipi","yakit_tipi_str",
            "motor_gucu","vites_oto","tramer","motor_hacmi_parsed"]].copy()
    rows.append(t.drop_duplicates(subset=["km","yil","fiyat","marka"]))

if len(sah_26):
    s = sah_26.copy()
    s["model_tam"] = s["seri"].fillna("")+" "+s["model"].fillna("")
    s["km"]        = s["kilometre"]
    s["sehir"]     = s["konum"].fillna("Bilinmiyor").str.split(",").str[-1].str.strip()
    s["satici_tipi"] = "Bilinmiyor"
    s["renk"]      = "Bilinmiyor"
    s["kaynak"]    = "sahibinden_2026"
    s["kasa_tipi"] = s["kasa_tipi"].replace("-","Bilinmiyor").fillna("Bilinmiyor")
    s["yakit_tipi_str"] = s["yakit_tipi"].fillna("Bilinmiyor")
    s["motor_hacmi_parsed"] = s["motor_hacmi"].fillna(0) / 1000
    s["motor_gucu"] = pd.to_numeric(s["motor_gucu"], errors="coerce").fillna(0)
    s["vites_oto"] = s["vites_tipi"].fillna("Manuel").str.lower().isin(["otomatik","yarı otomatik","cvt"]).astype(int)
    s["tramer"]  = pd.to_numeric(s["tramer"], errors="coerce").fillna(0)
    s["degisen"] = pd.to_numeric(s["degisen"], errors="coerce").fillna(0)
    s["boyali"]  = pd.to_numeric(s["boyali"],  errors="coerce").fillna(0)
    t = s[["marka","model_tam","yil","km","fiyat","renk","sehir","satici_tipi",
            "kaynak","degisen","boyali","kasa_tipi","yakit_tipi_str",
            "motor_gucu","vites_oto","tramer","motor_hacmi_parsed"]].copy()
    rows.append(t.drop_duplicates(subset=["km","yil","fiyat","marka"]))

df = pd.concat(rows, ignore_index=True)
KAYNAK_YIL = {"arabam_base":2024,"sahibinden_base":2024,
              "arabam_kaggle":2025,"sahibinden_kaggle":2025,"sahibinden_2026":2026}
df["kaynak_yil"] = df["kaynak"].map(KAYNAK_YIL).astype(float)

for c in ["degisen","boyali","motor_gucu","vites_oto","tramer"]:
    df[c] = df[c].fillna(0)
df["kasa_tipi"] = df["kasa_tipi"].fillna("Bilinmiyor")
df["yakit_tipi_str"] = df["yakit_tipi_str"].fillna("Bilinmiyor")

print("[2] Temizleniyor...")
df["fiyat_tl"] = df["fiyat"].apply(pf)
df["km_sayi"]  = df["km"].apply(pk)
df["yil"]      = pd.to_numeric(df["yil"], errors="coerce")
df = df.dropna(subset=["fiyat_tl","km_sayi","yil"])
df = df[(df["fiyat_tl"]>50_000)&(df["fiyat_tl"]<50_000_000)]
df = df[(df["km_sayi"]>=0)&(df["km_sayi"]<1_500_000)]
df = df[(df["yil"]>=1990)&(df["yil"]<=2026)]
keep = df.groupby("marka")["fiyat_tl"].transform(lambda x: (x>=x.quantile(0.01))&(x<=x.quantile(0.99)))
df = df[keep].reset_index(drop=True)
print(f"  {len(df):,} ilan")

print("[3] Özellikler...")
df["yas"]         = 2026 - df["yil"]
df["log_km"]      = np.log1p(df["km_sayi"])
df["km_per_year"] = df["km_sayi"] / df["yas"].clip(lower=1)
df["log_fiyat"]   = np.log1p(df["fiyat_tl"])

def extract_cc(m):
    r = re.search(r'\b(\d+\.\d+)\b', str(m))
    if r:
        v = float(r.group(1))
        if 0.5<=v<=8.0: return v
    return np.nan

def engine_type(m):
    ml = str(m).lower()
    if any(x in ml for x in ["elektrik","electric"]): return 3
    if any(x in ml for x in ["hibrit","hybrid"]): return 2
    if any(x in ml for x in ["tsi","tdi","cdti","cdi","hdi","dci","crdi"]): return 1
    return 0

df["motor_hacmi_model"] = df["model_tam"].apply(extract_cc)
df["motor_hacmi"] = df["motor_hacmi_model"].combine_first(df["motor_hacmi_parsed"])
df["motor_hacmi"] = df["motor_hacmi"].fillna(df["motor_hacmi"].median())
df["motor_tipi"]  = df["model_tam"].apply(engine_type)
df["donanim"]     = df["model_tam"].apply(donanim)
df["yakit_d"]     = df["yakit_tipi_str"].apply(yakit_enc)
df["yakit_m"]     = df["model_tam"].apply(yakit_enc)
df["yakit"]       = df["yakit_d"].where(df["yakit_d"]>0, df["yakit_m"])

df["power_per_liter"] = (df["motor_gucu"] / df["motor_hacmi"].clip(lower=0.1)).clip(upper=500)
df["yas_sq"]          = df["yas"] ** 2
df["log_km_yas"]      = df["log_km"] * df["yas"]

def kasa_enc(row):
    kt = str(row["kasa_tipi"]).lower().strip()
    v = KASA_MAP.get(kt)
    if v is not None: return v
    ml = str(row["model_tam"]).lower()
    if any(x in ml for x in ["hatchback","hb"]): return 2
    if "sedan" in ml: return 0
    if any(x in ml for x in [" suv","4x4"]): return 3
    if "crossover" in ml: return 4
    if any(x in ml for x in ["station","touring","avant","variant"]): return 6
    if any(x in ml for x in ["coupe","coupé"]): return 9
    return np.nan

df["kasa_enc"]   = df.apply(kasa_enc, axis=1).astype(float)
df["degisen"]    = df["degisen"].clip(0,20)
df["boyali"]     = df["boyali"].clip(0,20)
df["kaza_var"]   = (df["degisen"]>0).astype(int)
df["boyali_var"] = (df["boyali"]>2).astype(int)
df["log_tramer"] = np.log1p(df["tramer"])

for c in ["marka","renk","sehir","satici_tipi","model_tam"]:
    df[c] = df[c].fillna("Bilinmiyor").astype(str).str.strip()

print("[4] Encoding...")
encoders = {}
for c in ["marka","renk","sehir","satici_tipi"]:
    le = LabelEncoder()
    df[c+"_enc"] = le.fit_transform(df[c])
    encoders[c] = le

df["model_freq"] = df["model_tam"].map(df["model_tam"].value_counts())
df["marka_cc"]   = df["marka_enc"] * df["motor_hacmi"]

def target_encode(df, col, target="log_fiyat", n_folds=5, s=10):
    gm = df[target].mean()
    result = pd.Series(gm, index=df.index)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr, va in kf.split(df):
        stats = df.iloc[tr].groupby(col)[target].agg(["mean","count"])
        smooth = (stats["count"]*stats["mean"]+s*gm)/(stats["count"]+s)
        result.iloc[va] = df.iloc[va][col].map(smooth).fillna(gm)
    return result

df["marka_target"] = target_encode(df, "marka")
df["model_target"] = target_encode(df, "model_tam")

FEATURES = [
    "yas","log_km","km_per_year",
    "motor_hacmi","motor_tipi","motor_gucu",
    "donanim","yakit",
    "model_freq","marka_target","model_target",
    "marka_enc","renk_enc","sehir_enc","satici_tipi_enc",
    "degisen","boyali","kaza_var","boyali_var",
    "kasa_enc","vites_oto","log_tramer","kaynak_yil","marka_cc",
    "power_per_liter","yas_sq","log_km_yas",
]

X = df[FEATURES]; y = df["log_fiyat"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

LGB_PARAMS = dict(
    max_depth=10, num_leaves=255, learning_rate=0.025,
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=25, reg_alpha=0.05, reg_lambda=0.8,
    random_state=42, n_jobs=-1, verbosity=-1,
)

CHECKPOINT = "carswipe_checkpoint.pkl"

if PHASE == 1:
    print("[5] Aşama 1: 3500 iter...")
    model = lgb.LGBMRegressor(n_estimators=3500, **LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)],
    )
    joblib.dump((model, encoders, FEATURES, X_train, X_test, y_train, y_test), CHECKPOINT)
    print(f"  Checkpoint kaydedildi. Best iter: {model.best_iteration_}")
else:
    print("[5] Aşama 2: checkpoint'ten devam (3500 iter daha)...")
    model_prev, encoders, FEATURES, X_train, X_test, y_train, y_test = joblib.load(CHECKPOINT)
    # LightGBM init_model ile devam et
    booster = model_prev.booster_
    model2 = lgb.LGBMRegressor(n_estimators=3500, **LGB_PARAMS)
    model2.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        init_model=booster,
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)],
    )
    model = model2
    print(f"  Toplam ağaç: {model.best_iteration_}")

print("[6] Sonuçlar...")
yp = np.expm1(model.predict(X_test))
yt = np.expm1(y_test)
mape   = np.mean(np.abs((yt-yp)/yt))*100
medape = np.median(np.abs((yt-yp)/yt))*100
r2     = r2_score(yt, yp)
err    = np.abs((yt-yp)/yt)*100
print(f"  MAPE: {mape:.1f}% | MedAPE: {medape:.1f}% | R²: {r2:.4f}")
print(f"  <10%: %{(err<10).mean()*100:.0f} | <20%: %{(err<20).mean()*100:.0f} | <30%: %{(err<30).mean()*100:.0f}")

fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n  Top 12 özellik:")
for feat, imp in fi.head(12).items():
    print(f"    {feat:28s} {imp:.0f}")

if PHASE == 2:
    print("\n[7] Final model kaydediliyor...")
    joblib.dump(model,    "carswipe_model.pkl")
    joblib.dump(encoders, "carswipe_encoders.pkl")
    joblib.dump(FEATURES, "carswipe_features.pkl")
    joblib.dump(sorted(df["marka"].unique().tolist()), "carswipe_markalar.pkl")
    joblib.dump("lgb",    "carswipe_model_type.pkl")
    joblib.dump(True,     "carswipe_has_extra.pkl")
    joblib.dump(KASA_MAP, "carswipe_kasa_map.pkl")
    gm = df["log_fiyat"].mean()
    ms = df.groupby("marka")["log_fiyat"].agg(["mean","count"])
    mt = df.groupby("model_tam")["log_fiyat"].agg(["mean","count"])
    joblib.dump((ms, mt, gm, 10), "carswipe_target_enc.pkl")
    print(f"  ✓ Final model kaydedildi! MAPE: {mape:.1f}%")
