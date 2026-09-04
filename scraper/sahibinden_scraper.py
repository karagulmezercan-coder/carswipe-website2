"""
sahibinden.com İlan Scraper — Gerçek Chrome Profili ile
Cloudflare bypass için kullanıcının kendi Chrome profilini kullanır.

⚠️  Çalıştırmadan önce Chrome'u tamamen kapat!

Kurulum: pip install selenium beautifulsoup4
Kullanım: python sahibinden_scraper.py
"""

import undetected_chromedriver as uc
from bs4 import BeautifulSoup
import csv, time, random, os, subprocess
from datetime import datetime

# ── Ayarlar ──────────────────────────────────────────────
OUTPUT_FILE   = "sahibinden_ilanlar.csv"
PROGRESS_FILE = "sahibinden_progress.txt"
DELAY_MIN     = 2.0
DELAY_MAX     = 4.0
PAGE_SIZE     = 50

# Gerçek Chrome profil yolu (Chrome kapalı olmalı!)
CHROME_PROFILE_DIR = r"C:\Users\erkan\AppData\Local\Google\Chrome\User Data"
CHROME_PROFILE     = "Default"   # Farklı profil varsa "Profile 1" gibi değiştir
# ─────────────────────────────────────────────────────────

# sahibinden.com marka slug'ları
MARKALAR = [
    ("Abarth",       "abarth"),
    ("Alfa Romeo",   "alfa-romeo"),
    ("Aston Martin", "aston-martin"),
    ("Audi",         "audi"),
    ("Bentley",      "bentley"),
    ("BMW",          "bmw"),
    ("Cadillac",     "cadillac"),
    ("Chevrolet",    "chevrolet"),
    ("Chrysler",     "chrysler"),
    ("Citroen",      "citroen"),
    ("Cupra",        "cupra"),
    ("Dacia",        "dacia"),
    ("Daewoo",       "daewoo"),
    ("Daihatsu",     "daihatsu"),
    ("Dodge",        "dodge"),
    ("DS",           "ds"),
    ("Fiat",         "fiat"),
    ("Ford",         "ford"),
    ("Honda",        "honda"),
    ("Hyundai",      "hyundai"),
    ("Infiniti",     "infiniti"),
    ("Isuzu",        "isuzu"),
    ("Jaguar",       "jaguar"),
    ("Jeep",         "jeep"),
    ("Kia",          "kia"),
    ("Lada",         "lada"),
    ("Lamborghini",  "lamborghini"),
    ("Land Rover",   "land-rover"),
    ("Lexus",        "lexus"),
    ("Maserati",     "maserati"),
    ("Mazda",        "mazda"),
    ("Mercedes-Benz","mercedes-benz"),
    ("MG",           "mg"),
    ("Mini",         "mini"),
    ("Mitsubishi",   "mitsubishi"),
    ("Nissan",       "nissan"),
    ("Opel",         "opel"),
    ("Peugeot",      "peugeot"),
    ("Pontiac",      "pontiac"),
    ("Porsche",      "porsche"),
    ("Renault",      "renault"),
    ("Rolls-Royce",  "rolls-royce"),
    ("Saab",         "saab"),
    ("Seat",         "seat"),
    ("Skoda",        "skoda"),
    ("Smart",        "smart"),
    ("Ssangyong",    "ssangyong"),
    ("Subaru",       "subaru"),
    ("Suzuki",       "suzuki"),
    ("Tesla",        "tesla"),
    ("TOGG",         "togg"),
    ("Toyota",       "toyota"),
    ("Volkswagen",   "volkswagen"),
    ("Volvo",        "volvo"),
    ("BYD",          "byd"),
    ("Chery",        "chery"),
    ("Geely",        "geely"),
    ("Leapmotor",    "leapmotor"),
    ("Tofas",        "tofas"),
    ("Volta",        "volta"),
]

CSV_FIELDS = [
    "marka","model_tam","yil","km","fiyat",
    "renk","sehir","ilce","satici_tipi","ilan_tarihi","url"
]

def get_driver():
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1280,900")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    driver = uc.Chrome(options=options, headless=False)
    return driver

def make_url(slug, offset):
    return (
        f"https://www.sahibinden.com/{slug}"
        f"?pagingOffset={offset}&pagingSize={PAGE_SIZE}"
    )

def wait_for_listings(driver, timeout=60, debug_file=None):
    """Sayfa yüklenene kadar bekle. CF challenge varsa kullanıcı çözsün."""
    cf_warned = False
    end = time.time() + timeout
    while time.time() < end:
        src = driver.page_source
        if "searchResultsItem" in src:
            return True
        if "cf-browser-verification" in src or "Just a moment" in src or "Checking your browser" in src:
            if not cf_warned:
                print("\n  ⚠ Cloudflare challenge! Açık Chrome penceresinde varsa tıkla/çöz. Bekleniyor (60s)...")
                cf_warned = True
            time.sleep(3)
            continue
        if "sign" in src.lower() and "password" in src.lower():
            print("  [⚠ Login sayfası! Sahibinden'e giriş yapman gerekiyor.]")
            if debug_file:
                with open(debug_file, "w", encoding="utf-8") as df:
                    df.write(src)
            return False
        time.sleep(1)
    if debug_file:
        with open(debug_file, "w", encoding="utf-8") as df:
            df.write(driver.page_source)
    return False

def parse_page(soup, marka_adi):
    rows = []
    for tr in soup.select("tr.searchResultsItem"):
        try:
            # URL + başlık
            link_el = tr.select_one("a.classifiedTitle")
            if not link_el:
                continue
            baslik = link_el.get("title", link_el.get_text(strip=True))
            href   = link_el.get("href", "")
            url    = "https://www.sahibinden.com" + href if href.startswith("/") else href

            # Seri + Model (iki ayrı td.searchResultsTagAttributeValue)
            tag_tds = tr.select("td.searchResultsTagAttributeValue")
            seri  = tag_tds[0].get_text(strip=True) if len(tag_tds) > 0 else ""
            model = tag_tds[1].get_text(strip=True) if len(tag_tds) > 1 else ""
            model_tam = f"{marka_adi} {seri} {model}".strip()

            # Yıl, KM, Renk (td.searchResultsAttributeValue sırasıyla)
            attr_tds = tr.select("td.searchResultsAttributeValue")
            yil  = attr_tds[0].get_text(strip=True) if len(attr_tds) > 0 else ""
            km   = attr_tds[1].get_text(strip=True) if len(attr_tds) > 1 else ""
            renk = attr_tds[2].get_text(strip=True) if len(attr_tds) > 2 else ""

            # Fiyat
            fiyat_el = tr.select_one("td.searchResultsPriceValue span")
            fiyat = fiyat_el.get_text(strip=True) if fiyat_el else ""

            # Tarih
            tarih_el = tr.select_one("td.searchResultsDateValue")
            tarih = " ".join(tarih_el.get_text(separator=" ").split()) if tarih_el else ""

            # Şehir / İlçe
            sehir, ilce = "", ""
            loc_el = tr.select_one("td.searchResultsLocationValue")
            if loc_el:
                parts = [s.strip() for s in loc_el.get_text(separator="\n").split("\n") if s.strip()]
                sehir = parts[0] if parts else ""
                ilce  = parts[1] if len(parts) > 1 else ""

            # Satıcı tipi: galeri ilanlarında store-icon linki olur
            satici = "Sahibinden"
            if tr.select_one("a.store-icon"):
                satici = "Galeriden"

            if not url:
                continue

            rows.append({
                "marka":       marka_adi,
                "model_tam":   model_tam,
                "yil":         yil,
                "km":          km,
                "fiyat":       fiyat,
                "renk":        renk,
                "sehir":       sehir,
                "ilce":        ilce,
                "satici_tipi": satici,
                "ilan_tarihi": tarih,
                "url":         url,
            })
        except Exception:
            continue
    return rows

def main():
    print("=" * 60)
    print("sahibinden.com İlan Scraper — Gerçek Chrome Profili")
    print(f"Başlangıç: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Toplam marka: {len(MARKALAR)}")
    print()
    print("⚠️  Chrome'un tamamen kapalı olduğundan emin ol!")
    print("=" * 60)

    # Mevcut veriler
    existing_urls = set()
    done_markalar = set()
    write_mode = "w"

    if os.path.exists(OUTPUT_FILE):
        ans = input(f"\n'{OUTPUT_FILE}' zaten var. Devam et (d) / Yeniden başla (y)? ").strip().lower()
        if ans == "d":
            write_mode = "a"
            with open(OUTPUT_FILE, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_urls.add(row.get("url", ""))
            print(f"  Mevcut {len(existing_urls):,} ilan yüklendi.")
            if os.path.exists(PROGRESS_FILE):
                with open(PROGRESS_FILE, "r", encoding="utf-8") as pf:
                    done_markalar = set(line.strip() for line in pf if line.strip())
                print(f"  Tamamlanan marka: {len(done_markalar)} → atlanacak.")
        else:
            if os.path.exists(PROGRESS_FILE):
                os.remove(PROGRESS_FILE)

    print("\nChrome başlatılıyor...")
    driver = get_driver()

    # Kullanıcı sahibinden'e giriş yapsın
    print("\n" + "=" * 60)
    print("Adım 1: Açılan Chrome penceresinde sahibinden.com'a giriş yap.")
    print("Adım 2: Giriş yapınca buraya dön ve Enter'a bas.")
    print("=" * 60)
    driver.get("https://www.sahibinden.com/giris")
    input("\nGiriş yaptıktan sonra Enter'a bas: ")

    total_saved = len(existing_urls)

    try:
        with open(OUTPUT_FILE, write_mode, newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_mode == "w":
                writer.writeheader()

            for m_idx, (marka_adi, slug) in enumerate(MARKALAR, 1):
                if slug in done_markalar:
                    print(f"\n[{m_idx}/{len(MARKALAR)}] {marka_adi} ✓ atlandı")
                    continue

                print(f"\n[{m_idx}/{len(MARKALAR)}] {marka_adi.upper()} ", end="", flush=True)

                marka_saved = 0
                offset = 0
                consec_empty = 0
                consec_fail  = 0

                while True:
                    url = make_url(slug, offset)
                    try:
                        driver.get(url)
                        debug_f = "debug_page.html" if (m_idx == 1 and offset == 0) else None
                        loaded = wait_for_listings(driver, timeout=20, debug_file=debug_f)
                    except Exception as e:
                        consec_fail += 1
                        print(f"  offset{offset}:⚠({consec_fail}) ", end="", flush=True)
                        if consec_fail >= 3:
                            print(f"\n  ⟳ Driver yenileniyor...", flush=True)
                            try: driver.quit()
                            except: pass
                            time.sleep(random.uniform(10, 18))
                            driver = get_driver()
                            consec_fail = 0
                        else:
                            time.sleep(random.uniform(6, 10))
                        continue

                    if not loaded:
                        consec_empty += 1
                        print(f"  off{offset}:∅ ", end="", flush=True)
                        if consec_empty >= 2:
                            print(f"\n  ✂ Sayfa boş → marka bitti.", flush=True)
                            break
                        time.sleep(random.uniform(3, 5))
                        offset += PAGE_SIZE
                        continue

                    consec_fail = 0
                    raw_html = driver.page_source
                    # İlk yüklemede HTML'i kaydet
                    if m_idx == 1 and offset == 0:
                        with open("debug_page.html", "w", encoding="utf-8") as dbf:
                            dbf.write(raw_html)
                        print(f"\n  [debug_page.html kaydedildi]", flush=True)
                    soup  = BeautifulSoup(raw_html, "html.parser")
                    items = parse_page(soup, marka_adi)

                    saved = 0
                    for item in items:
                        if item["url"] and item["url"] not in existing_urls:
                            writer.writerow(item)
                            existing_urls.add(item["url"])
                            saved += 1
                    f.flush()

                    marka_saved  += saved
                    total_saved  += saved
                    print(f"  off{offset}:{saved} ", end="", flush=True)

                    if saved == 0:
                        consec_empty += 1
                        if consec_empty >= 3:
                            print(f"\n  ✂ 3 boş offset → marka bitti.", flush=True)
                            break
                    else:
                        consec_empty = 0

                    # sahibinden offset tabanlı sayfalama
                    offset += PAGE_SIZE
                    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

                print(f"\n  → {marka_adi.upper()} toplam: {marka_saved} ilan | Genel: {total_saved:,}")

                with open(PROGRESS_FILE, "a", encoding="utf-8") as pf:
                    pf.write(slug + "\n")
                done_markalar.add(slug)

                # Her markadan sonra kısa mola
                if m_idx < len(MARKALAR):
                    mola = random.uniform(5, 10)
                    print(f"  ⏳ {mola:.0f}s mola...", flush=True)
                    time.sleep(mola)

    except KeyboardInterrupt:
        print("\n\n⚠ Ctrl+C ile durduruldu. Kaydedilenler korundu.")
    finally:
        driver.quit()

    print("\n" + "=" * 60)
    print(f"✓ Tamamlandı! Toplam {total_saved:,} ilan")
    print(f"✓ Dosya: {OUTPUT_FILE}")
    print(f"Bitiş: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
