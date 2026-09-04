import pandas as pd, numpy as np, re, joblib, warnings, math
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

UPLOADS  = "/sessions/zealous-keen-planck/mnt/uploads"
SAVE_DIR = "/sessions/zealous-keen-planck/mnt/uygulama car/scraper"

def parse_hacmi(s):
    n=re.findall(r"\d+",str(s))
    if not n: return np.nan
    v=float(n[0]); return v/1000 if v>10 else v

def parse_gucu(s):
    n=re.findall(r"\d+",str(s).replace("HP",""))
    return np.mean([int(x) for x in n]) if n else np.nan

def yakit_enc(s):
    ml=str(s).lower()
    if any(x in ml for x in ["elektrik","electric"]): return 4
    if any(x in ml for x in ["hibrit","hybrid"]): return 3
    if "lpg" in ml: return 2
    if any(x in ml for x in ["dizel","diesel","tdi","cdi","hdi","dci"]): return 1
    return 0

DONANIM_MAP={"authentique":1,"ambiance":1,"access":1,"joy":2,"easy":2,"essential":2,"pop":2,
             "comfort":3,"touch":3,"trend":3,"techline":3,"core":3,"style":3,"feel":3,"plus":3,
             "elegance":4,"titanium":4,"highline":4,"prestige":4,"ambition":4,"signature":4,
             "sport":4,"dynamic":4,"gt":4,"fr":4,"premium":4,"s line":4,"sportline":4,
             "lounge":5,"r-line":5,"exclusive":5,"luxury":5,"amg":5,"m sport":5,"gti":5,"full":5}
def donanim(m): return max((v for k,v in DONANIM_MAP.items() if k in str(m).lower()),default=3)

KASA_MAP={"sedan":0,"hatchback/3":1,"hatchback/5":2,"suv":3,"crossover":4,"mpv":5,
          "station wagon":6,"pick-up":7,"cabriolet":8,"coupe":9,"roadster":10,"minivan":11,"panelvan":12}

def extract_cc(m):
    r=re.search(r"(\d+\.\d+)",str(m))
    if r:
        v=float(r.group(1))
        if 0.5<=v<=8.0: return v
    return np.nan

def engine_type(m):
    ml=str(m).lower()
    if any(x in ml for x in ["elektrik","electric"]): return 3
    if any(x in ml for x in ["hibrit","hybrid"]): return 2
    if any(x in ml for x in ["tsi","tdi","cdti","cdi","hdi","dci","crdi"]): return 1
    return 0

rows=[]

# 1. car_price_prediction_turkce.csv
d2=pd.read_csv(f"{UPLOADS}/car_price_prediction_turkce.csv",encoding="utf-8-sig")
d2["model_tam"]=d2["seri"].fillna("")+" "+d2["model"].fillna("")
d2["km"]=pd.to_numeric(d2["kilometre"],errors="coerce")
d2["fiyat_tl"]=pd.to_numeric(d2["fiyat"],errors="coerce")
d2["motor_h"]=d2["motor_hacmi"].apply(parse_hacmi)
d2["motor_g"]=d2["motor_gucu"].apply(parse_gucu)
d2["vites_oto"]=d2["vites_tipi"].fillna("Manuel").str.lower().isin(["otomatik","yari otomatik","cvt"]).astype(int)
d2["degisen"]=pd.to_numeric(d2["degisen_sayisi"],errors="coerce").fillna(0)
d2["boyali"]=pd.to_numeric(d2["boyali_sayisi"],errors="coerce").fillna(0)
d2["tramer"]=0.0; d2["sehir"]="Bilinmiyor"
d2["satici_tipi"]=d2["kimden"].fillna("Bilinmiyor")
t2=d2[["marka","model_tam","yil","km","fiyat_tl","renk","sehir","satici_tipi",
        "degisen","boyali","kasa_tipi","yakit_tipi","motor_g","vites_oto","tramer","motor_h"]].copy()
t2.columns=["marka","model_tam","yil","km","fiyat_tl","renk","sehir","satici_tipi",
            "degisen","boyali","kasa_tipi","yakit_tipi_str","motor_gucu","vites_oto","tramer","motor_hacmi_p"]
rows.append(t2); print(f"car_price_pred: {len(t2):,}")

# 2. cars1_turkce.csv
d3=pd.read_csv(f"{UPLOADS}/cars1_turkce.csv",encoding="utf-8-sig")
d3["model_tam"]=d3["seri"].fillna("")+" "+d3["model"].fillna("")
d3["km"]=pd.to_numeric(d3["kilometre"],errors="coerce")
d3["fiyat_tl"]=pd.to_numeric(d3["fiyat"],errors="coerce")
d3["motor_h"]=d3["motor_hacmi"].apply(parse_hacmi)
d3["motor_g"]=pd.to_numeric(d3["motor_gucu"],errors="coerce").fillna(0)
d3["vites_oto"]=d3["vites_tipi"].fillna("Manuel").str.lower().isin(["otomatik","yari otomatik","cvt"]).astype(int)
d3["degisen"]=pd.to_numeric(d3["degisen"],errors="coerce").fillna(0)
d3["boyali"]=pd.to_numeric(d3["boyali"],errors="coerce").fillna(0)
d3["tramer"]=pd.to_numeric(d3["tramer"],errors="coerce").fillna(0)
d3["renk"]="Bilinmiyor"; d3["sehir"]=d3["konum"].fillna("Bilinmiyor").str.strip()
d3["satici_tipi"]="Bilinmiyor"
t3=d3[["marka","model_tam","yil","km","fiyat_tl","renk","sehir","satici_tipi",
        "degisen","boyali","kasa_tipi","yakit_tipi","motor_g","vites_oto","tramer","motor_h"]].copy()
t3.columns=["marka","model_tam","yil","km","fiyat_tl","renk","sehir","satici_tipi",
            "degisen","boyali","kasa_tipi","yakit_tipi_str","motor_gucu","vites_oto","tramer","motor_hacmi_p"]
rows.append(t3); print(f"cars1: {len(t3):,}")

# 3. 1car_prices_tr_turkce.csv
d1=pd.read_csv(f"{UPLOADS}/1car_prices_tr_turkce.csv",encoding="utf-8-sig")
d1["model_tam"]=d1["seri"].fillna("")+" "+d1["model"].fillna("")
d1["fiyat_tl"]=pd.to_numeric(d1["fiyat(TRY)"].astype(str).str.replace(".","",regex=False).str.replace(",",".",regex=False),errors="coerce")
d1["km"]=pd.to_numeric(d1["Km"].astype(str).str.replace(".","",regex=False).str.replace(",",".",regex=False),errors="coerce")
d1["motor_h"]=d1["motorHacmi(Cc)"].apply(parse_hacmi)
d1["motor_g"]=d1["motorGucu(HP)"].apply(parse_gucu)
d1["vites_oto"]=d1["vitesTipi"].fillna("Manuel").str.lower().isin(["otomatik","yari otomatik","cvt"]).astype(int)
degisen_col=[c for c in d1.columns if "degi" in c.lower() and "par" in c.lower()]
boyali_col=[c for c in d1.columns if "boyal" in c.lower() and "par" in c.lower() and "lokal" not in c.lower()]
d1["degisen"]=pd.to_numeric(d1[degisen_col[0]],errors="coerce").fillna(0) if degisen_col else 0.0
d1["boyali"]=pd.to_numeric(d1[boyali_col[0]],errors="coerce").fillna(0) if boyali_col else 0.0
d1["tramer"]=0.0; d1["sehir"]=d1["il"].fillna("Bilinmiyor")
d1["satici_tipi"]=d1["saticiTuru"].fillna("Bilinmiyor")
d1["yil"]=pd.to_numeric(d1["y?l"],errors="coerce")
t1=d1[["marka","model_tam","yil","km","fiyat_tl","renk","sehir","satici_tipi",
        "degisen","boyali","kasaTipi","yakitTuru","motor_g","vites_oto","tramer","motor_h"]].copy()
t1.columns=["marka","model_tam","yil","km","fiyat_tl","renk","sehir","satici_tipi",
            "degisen","boyali","kasa_tipi","yakit_tipi_str","motor_gucu","vites_oto","tramer","motor_hacmi_p"]
rows.append(t1); print(f"1car_prices: {len(t1):,}")

# Birlestir & Temizle
df=pd.concat(rows,ignore_index=True)
for c in ["yil","km","fiyat_tl"]: df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.dropna(subset=["fiyat_tl","km","yil"])
df=df[(df["fiyat_tl"]>50000)&(df["fiyat_tl"]<10000000)]
df=df[(df["km"]>=0)&(df["km"]<1500000)]
df=df[(df["yil"]>=1990)&(df["yil"]<=2026)]
keep=df.groupby("marka")["fiyat_tl"].transform(lambda x:(x>=x.quantile(0.01))&(x<=x.quantile(0.99)))
df=df[keep].reset_index(drop=True)
print(f"Temiz toplam: {len(df):,}  Fiyat medyan: TL{df.fiyat_tl.median():,.0f}")

df["yas"]=2025-df["yil"]
df["log_km"]=np.log1p(df["km"])
df["km_per_year"]=df["km"]/df["yas"].clip(lower=1)
df["log_fiyat"]=np.log1p(df["fiyat_tl"])
df["motor_hacmi_model"]=df["model_tam"].apply(extract_cc)
df["motor_hacmi"]=df["motor_hacmi_model"].combine_first(df["motor_hacmi_p"])
df["motor_hacmi"]=df["motor_hacmi"].fillna(df["motor_hacmi"].median())
df["motor_tipi"]=df["model_tam"].apply(engine_type)
df["donanim"]=df["model_tam"].apply(donanim)
df["yakit_d"]=df["yakit_tipi_str"].apply(yakit_enc)
df["yakit_m"]=df["model_tam"].apply(yakit_enc)
df["yakit"]=df["yakit_d"].where(df["yakit_d"]>0,df["yakit_m"])
df["motor_gucu"]=pd.to_numeric(df["motor_gucu"],errors="coerce").fillna(0)
df["power_per_liter"]=(df["motor_gucu"]/df["motor_hacmi"].clip(lower=0.1)).clip(upper=500)
df["yas_sq"]=df["yas"]**2; df["log_km_yas"]=df["log_km"]*df["yas"]

def kasa_enc(row):
    kt=str(row["kasa_tipi"]).lower().strip()
    v=KASA_MAP.get(kt)
    if v is not None: return v
    ml=str(row["model_tam"]).lower()
    if any(x in ml for x in ["hatchback","hb"]): return 2
    if "sedan" in ml: return 0
    if any(x in ml for x in [" suv","4x4"]): return 3
    if "crossover" in ml: return 4
    return np.nan

df["kasa_enc"]=df.apply(kasa_enc,axis=1).astype(float)
df["degisen"]=pd.to_numeric(df["degisen"],errors="coerce").fillna(0).clip(0,20)
df["boyali"]=pd.to_numeric(df["boyali"],errors="coerce").fillna(0).clip(0,20)
df["kaza_var"]=(df["degisen"]>0).astype(int)
df["boyali_var"]=(df["boyali"]>2).astype(int)
df["log_tramer"]=np.log1p(pd.to_numeric(df["tramer"],errors="coerce").fillna(0))
df["vites_oto"]=pd.to_numeric(df["vites_oto"],errors="coerce").fillna(0).astype(int)
for c in ["marka","renk","sehir","satici_tipi"]: df[c]=df[c].fillna("Bilinmiyor").astype(str).str.strip()

encoders={}
for c in ["marka","renk","sehir","satici_tipi"]:
    le=LabelEncoder(); df[c+"_enc"]=le.fit_transform(df[c]); encoders[c]=le

df["model_freq"]=df["model_tam"].map(df["model_tam"].value_counts())
df["marka_cc"]=df["marka_enc"]*df["motor_hacmi"]

def target_encode(df,col,target="log_fiyat",n_folds=5,s=10):
    gm=df[target].mean(); result=pd.Series(gm,index=df.index)
    kf=KFold(n_splits=n_folds,shuffle=True,random_state=42)
    for tr,va in kf.split(df):
        stats=df.iloc[tr].groupby(col)[target].agg(["mean","count"])
        smooth=(stats["count"]*stats["mean"]+s*gm)/(stats["count"]+s)
        result.iloc[va]=df.iloc[va][col].map(smooth).fillna(gm)
    return result

df["marka_target"]=target_encode(df,"marka")
df["model_target"]=target_encode(df,"model_tam")

FEATURES=["yas","log_km","km_per_year","motor_hacmi","motor_tipi","motor_gucu","donanim","yakit",
          "model_freq","marka_target","model_target","marka_enc","renk_enc","sehir_enc","satici_tipi_enc",
          "degisen","boyali","kaza_var","boyali_var","kasa_enc","vites_oto","log_tramer","marka_cc",
          "power_per_liter","yas_sq","log_km_yas"]

X=df[FEATURES]; y=df["log_fiyat"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=42)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

model=lgb.LGBMRegressor(n_estimators=8000,max_depth=10,num_leaves=255,learning_rate=0.05,
                         subsample=0.8,colsample_bytree=0.8,min_child_samples=25,
                         reg_alpha=0.05,reg_lambda=0.8,random_state=42,n_jobs=-1,verbosity=-1)
model.fit(X_train,y_train,eval_set=[(X_test,y_test)],
          callbacks=[lgb.early_stopping(200,verbose=False),lgb.log_evaluation(1000)])

yp=np.expm1(model.predict(X_test)); yt=np.expm1(y_test)
mape=np.mean(np.abs((yt-yp)/yt))*100
medape=np.median(np.abs((yt-yp)/yt))*100
r2=r2_score(yt,yp); err=np.abs((yt-yp)/yt)*100
print(f"MAPE:{mape:.1f}% MedAPE:{medape:.1f}% R2:{r2:.4f} best_iter:{model.best_iteration_}")
print(f"<10%:{(err<10).mean()*100:.0f}% <20%:{(err<20).mean()*100:.0f}%")

gm_global=df["log_fiyat"].mean()
ms_stats=df.groupby("marka")["log_fiyat"].agg(["mean","count"])
smooth_m=(ms_stats["count"]*ms_stats["mean"]+10*gm_global)/(ms_stats["count"]+10)
mt_stats=df.groupby("model_tam")["log_fiyat"].agg(["mean","count"])
smooth_mt=(mt_stats["count"]*mt_stats["mean"]+10*gm_global)/(mt_stats["count"]+10)

def quick_pred(marka,model_tam,yil,km,motor_g=0,vites_oto=0):
    yas=2025-yil; lkm=math.log1p(km)
    mh=extract_cc(model_tam)
    if isinstance(mh,float) and np.isnan(mh): mh=1.6
    me=int(encoders["marka"].transform([marka])[0]) if marka in list(encoders["marka"].classes_) else 0
    mt=float(smooth_mt.get(model_tam,float(smooth_m.get(marka,gm_global))))
    mr=float(smooth_m.get(marka,gm_global))
    mf=int(df["model_tam"].value_counts().get(model_tam,1))
    fm={"yas":yas,"log_km":lkm,"km_per_year":km/max(yas,1),"motor_hacmi":mh,
        "motor_tipi":engine_type(model_tam),"motor_gucu":motor_g,"donanim":donanim(model_tam),
        "yakit":yakit_enc(model_tam),"model_freq":mf,"marka_target":mr,"model_target":mt,
        "marka_enc":me,"renk_enc":0,"sehir_enc":0,"satici_tipi_enc":0,
        "degisen":0.0,"boyali":0.0,"kaza_var":0,"boyali_var":0,"kasa_enc":np.nan,
        "vites_oto":vites_oto,"log_tramer":0.0,"marka_cc":me*mh,
        "power_per_liter":motor_g/mh if mh else 0,"yas_sq":yas**2,"log_km_yas":lkm*yas}
    return math.expm1(float(model.predict(pd.DataFrame([fm],columns=FEATURES))[0]))

print('--- Kontrol Tahminleri ---')
print('2020 Clio 1.0 TCe 85k     -> TL%s' % format(int(quick_pred('Renault','Clio 1.0 TCe Touch',2020,85000,90,0)),','))
print('2021 BMW 320i 55k          -> TL%s' % format(int(quick_pred('BMW','3 Serisi 320i M Sport',2021,55000,184,1)),','))
print('2019 Egea 1.3 95k          -> TL%s' % format(int(quick_pred('Fiat','Egea 1.3 Multijet Easy',2019,95000,95,0)),','))
print('2018 Golf 1.6 TDI 120k     -> TL%s' % format(int(quick_pred('Volkswagen','Golf 1.6 TDI Comfortline',2018,120000,115,0)),','))

joblib.dump(model,f"{SAVE_DIR}/carswipe_model.pkl")
joblib.dump(encoders,f"{SAVE_DIR}/carswipe_encoders.pkl")
joblib.dump(FEATURES,f"{SAVE_DIR}/carswipe_features.pkl")
joblib.dump(sorted(df["marka"].unique().tolist()),f"{SAVE_DIR}/carswipe_markalar.pkl")
joblib.dump("lgb",f"{SAVE_DIR}/carswipe_model_type.pkl")
joblib.dump(True,f"{SAVE_DIR}/carswipe_has_extra.pkl")
joblib.dump(KASA_MAP,f"{SAVE_DIR}/carswipe_kasa_map.pkl")
joblib.dump((ms_stats,mt_stats,gm_global,10),f"{SAVE_DIR}/carswipe_target_enc.pkl")
print("Tum dosyalar kaydedildi!")
