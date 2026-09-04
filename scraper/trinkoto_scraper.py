"""
Trinkoto Fiyat Cache Scraper
============================
Trinkoto /araba-degeri/{marka}/{model} sayfalarını çekip
fiyat bandlarını trinkoto_cache.json dosyasına kaydeder.

Kullanım:
    python trinkoto_scraper.py            # Tüm markalar/modeller
    python trinkoto_scraper.py --test     # Sadece 5 popüler model

Cache dosyası api.py tarafından /validate endpoint'inde kullanılır.

NOT: Trinkoto bazı IP'leri bloklayabilir (Cloudflare).
     VPN veya residential proxy gerekebilir.
     Alternatif: cloudscraper kütüphanesi.
"""
import re, json, time, argparse, os
from pathlib import Path

try:
    import cloudscraper
    SESSION = cloudscraper.create_scraper()
    print("✓ cloudscraper kullanılıyor (Cloudflare bypass)")
except ImportError:
    import requests
    SESSION = requests.Session()
    SESSION.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "tr-TR,tr;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    print("! cloudscraper yok, requests kullanılıyor")
    print("  403 alırsanız: pip install cloudscraper")

TRINKOTO_BASE = "https://www.trinkoto.com/araba-degeri"
CACHE_FILE    = Path(__file__).parent / "trinkoto_cache.json"
DELAY_SEC     = 1.5  #礼貌的 rate limiting

# Trinkoto ana sayfasından çıkarılan 38 marka
ALL_BRANDS = [
    "volkswagen","renault","fiat","ford","opel","peugeot","hyundai","citroen",
    "toyota","mercedes-benz","bmw","audi","skoda","nissan","honda","kia","seat",
    "dacia","chery","volvo","chevrolet","land-rover","kgm-ssangyong","cupra",
    "alfa-romeo","mg","jeep","tesla","mitsubishi","mini","suzuki","ds-automobiles",
    "ssangyong","porsche","mazda","subaru","jaguar","maserati",
]

# Popüler modeller (--test modu için)
TEST_MODELS = [
    ("renault",     "clio"),
    ("volkswagen",  "golf"),
    ("fiat",        "egea"),
    ("toyota",      "corolla"),
    ("hyundai",     "i20"),
]

def slug(s: str) -> str:
    return re.sub(r'\s+', '-', s.strip().lower())

def parse_prices(html: str) -> dict | None:
    """HTML'den versiyon fiyat tablosunu çıkar."""
    prices = re.findall(r'₺([\d.]+)(?:\s*[—–-]\s*₺([\d.]+))?', html)
    if not prices:
        return None
    vals = []
    for lo, hi in prices:
        vals.append(int(lo.replace('.', '')))
        if hi:
            vals.append(int(hi.replace('.', '')))
    if not vals:
        return None

    # Versiyon bazlı satırları da çıkar
    versiyonlar = []
    # Motor + donanım + fiyat örüntüsü
    pattern = r'>\s*([\d.,]+\s*(?:TCe|TDI|SCe|dCi|E-Tech|Multijet|TFSI|TSI|CDTi|HDi|CRDi|D|Elektrik|Hibrit)[^<]*?)<.*?₺([\d.]+)(?:\s*[—–-]\s*₺([\d.]+))?'
    for m in re.finditer(pattern, html, re.IGNORECASE):
        motor = m.group(1).strip()[:50]
        lo = int(m.group(2).replace('.', ''))
        hi = int(m.group(3).replace('.', '')) if m.group(3) else lo
        versiyonlar.append({"motor": motor, "min": lo, "max": hi, "ort": (lo+hi)//2})

    return {
        "min": min(vals),
        "max": max(vals),
        "ort": sum(vals) // len(vals),
        "n":   len(prices),
        "versiyonlar": versiyonlar[:20],  # İlk 20 versiyon
    }

def fetch_model(marka_slug: str, model_slug: str) -> dict | None:
    url = f"{TRINKOTO_BASE}/{marka_slug}/{model_slug}"
    try:
        r = SESSION.get(url, timeout=12)
        if r.status_code == 403:
            print(f"    403 Forbidden → {url}")
            return None
        if r.status_code != 200:
            print(f"    HTTP {r.status_code} → {url}")
            return None
        data = parse_prices(r.text)
        if data:
            data["url"] = url
        return data
    except Exception as e:
        print(f"    Hata: {e}")
        return None

def fetch_brand_models(marka_slug: str) -> list[str]:
    """Marka sayfasından model slug listesi çıkar."""
    url = f"{TRINKOTO_BASE}/{marka_slug}"
    try:
        r = SESSION.get(url, timeout=12)
        if r.status_code != 200:
            return []
        # /araba-degeri/{marka}/{model} linkleri
        models = re.findall(
            rf'/araba-degeri/{re.escape(marka_slug)}/([a-z0-9-]+)(?:/|")',
            r.text
        )
        return list(dict.fromkeys(models))  # deduplicate, sıra koru
    except Exception:
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Sadece 5 test modeli")
    parser.add_argument("--brand", help="Tek marka (ör. renault)")
    args = parser.parse_args()

    # Mevcut cache'i yükle (devam edebilmek için)
    cache = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        print(f"Mevcut cache: {len(cache)} kayıt")

    if args.test:
        pairs = TEST_MODELS
    elif args.brand:
        models = fetch_brand_models(args.brand)
        print(f"{args.brand}: {len(models)} model bulundu")
        pairs = [(args.brand, m) for m in models]
    else:
        pairs = []
        for brand in ALL_BRANDS:
            models = fetch_brand_models(brand)
            print(f"{brand}: {len(models)} model")
            pairs.extend([(brand, m) for m in models])
            time.sleep(DELAY_SEC)

    print(f"\nToplam {len(pairs)} model sayfası taranacak\n")
    ok = skip = err = 0

    for marka_slug, model_slug in pairs:
        key = f"{marka_slug}/{model_slug}"
        if key in cache:
            skip += 1
            continue
        print(f"  → {key} ... ", end="", flush=True)
        data = fetch_model(marka_slug, model_slug)
        if data:
            cache[key] = data
            print(f"₺{data['min']:,}–₺{data['max']:,} ({data['n']} fiyat)")
            ok += 1
        else:
            cache[key] = None
            print("veri yok")
            err += 1
        # Her 10 kayıtta bir kaydet
        if (ok + err) % 10 == 0:
            CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(DELAY_SEC)

    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Cache kaydedildi: {CACHE_FILE}")
    print(f"  OK: {ok} | Atlandı: {skip} | Hata/boş: {err}")

if __name__ == "__main__":
    main()
