"""
arabam.com İlan Scraper - Marka bazlı (tüm ilanlar)
Her markayı ayrı ayrı çekerek 50 sayfa sınırını aşar.

Kurulum: pip install selenium beautifulsoup4
Kullanım: python arabam_scraper.py
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import csv, time, random, os, re
from datetime import datetime

# ── Ayarlar ──────────────────────────────────────────────
OUTPUT_FILE   = "arabam_ilanlar.csv"
PROGRESS_FILE = "arabam_progress.txt"   # tamamlanan markalar
DELAY_MIN     = 1.5
DELAY_MAX     = 3.0
# ─────────────────────────────────────────────────────────

# Türkiye'deki başlıca markalar (arabam.com URL slug'ları)
MARKALAR = [
    "abarth","alfa-romeo","aston-martin","audi","bentley","bmw","bugatti",
    "cadillac","chevrolet","chrysler","citroen","cupra","dacia","daewoo",
    "daihatsu","dodge","ds","fiat","ford","honda","hummer","hyundai",
    "infiniti","isuzu","jaguar","jeep","kia","lada","lamborghini","lancia",
    "land-rover","lexus","lincoln","lotus","maserati","maybach","mazda",
    "mclaren","mercedes-benz","mg","mini","mitsubishi","nissan","opel",
    "peugeot","pontiac","porsche","renault","rolls-royce","saab","seat",
    "skoda","smart","ssangyong","subaru","suzuki","tesla","togg","toyota",
    "volkswagen","volvo",
]

CSV_FIELDS = [
    "marka","model_tam","yil","km","fiyat",
    "yakit","vites","renk","sehir","satici_tipi","ilan_tarihi","url"
]

def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=options)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver

def get_total_pages(driver, url):
    driver.get(url)
    time.sleep(2.5)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    pagination = soup.select(".pagination a")
    nums = [int(a.get_text(strip=True)) for a in pagination
            if a.get_text(strip=True).isdigit()]
    return max(nums) if nums else 1

def parse_listings(soup, marka_slug):
    rows = []
    for row in soup.select("tr.listing-list-item"):
        try:
            tds = row.find_all("td")

            model_el = row.select_one(".listing-modelname h2")
            model_tam = model_el.get_text(strip=True) if model_el else ""

            parts = model_tam.split(" ", 1)
            marka = parts[0] if parts else marka_slug.replace("-", " ").title()

            yil = tds[3].get_text(strip=True) if len(tds) > 3 else ""
            km  = tds[4].get_text(strip=True) if len(tds) > 4 else ""

            price_el = row.select_one(".listing-price")
            fiyat = price_el.get_text(strip=True) if price_el else ""

            renk = tds[5].get_text(strip=True) if len(tds) > 5 else ""

            sehir = ""
            loc_el = row.select_one(".listing-location span")
            if loc_el:
                sehir = loc_el.get_text(strip=True)

            satici = "Sahibinden"
            satici_el = row.select_one(".listing-text")
            if satici_el and "galeri" in satici_el.get_text().lower():
                satici = "Galeriden"

            tarih = tds[7].get_text(strip=True) if len(tds) > 7 else ""

            url = ""
            link_el = row.select_one("a[href*='/ilan/']")
            if link_el:
                href = link_el.get("href", "")
                url = "https://www.arabam.com" + href if href.startswith("/") else href

            # Yakıt/vites model adından
            yakit, vites = "", ""
            ml = model_tam.lower()
            for y in ["benzin","dizel","lpg","hibrit","elektrik","hybrid"]:
                if y in ml:
                    yakit = y.capitalize(); break
            for v in ["otomatik","manuel","yarı otomatik"]:
                if v in ml:
                    vites = v.capitalize(); break

            rows.append({
                "marka": marka, "model_tam": model_tam,
                "yil": yil, "km": km, "fiyat": fiyat,
                "yakit": yakit, "vites": vites, "renk": renk,
                "sehir": sehir, "satici_tipi": satici,
                "ilan_tarihi": tarih, "url": url,
            })
        except Exception:
            continue
    return rows

def main():
    print("=" * 55)
    print("arabam.com İlan Scraper — Marka bazlı")
    print(f"Başlangıç: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Toplam marka: {len(MARKALAR)}")
    print("=" * 55)

    # Kaldığı yerden devam için mevcut URL'leri yükle
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
            # Tamamlanan markaları yükle
            if os.path.exists(PROGRESS_FILE):
                with open(PROGRESS_FILE, "r", encoding="utf-8") as pf:
                    done_markalar = set(line.strip() for line in pf if line.strip())
                print(f"  Tamamlanan marka: {len(done_markalar)} → atlanacak.")
        else:
            write_mode = "w"
            # İlerleme dosyasını da sıfırla
            if os.path.exists(PROGRESS_FILE):
                os.remove(PROGRESS_FILE)

    print("\nChrome başlatılıyor...")
    driver = get_driver()
    total_saved = len(existing_urls)

    try:
        with open(OUTPUT_FILE, write_mode, newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_mode == "w":
                writer.writeheader()

            for m_idx, marka in enumerate(MARKALAR, 1):
                if marka in done_markalar:
                    print(f"\n[{m_idx}/{len(MARKALAR)}] {marka.upper()} ✓ atlandı")
                    continue

                base = f"https://www.arabam.com/ikinci-el/otomobil/{marka}?take=50&page={{page}}"
                print(f"\n[{m_idx}/{len(MARKALAR)}] {marka.upper()} ", end="", flush=True)

                try:
                    total_pages = get_total_pages(driver, base.format(page=1))
                except Exception:
                    print("⚠ Atlandı")
                    continue

                print(f"({total_pages} sayfa)", flush=True)
                marka_saved = 0
                consec_timeouts = 0  # arka arkaya timeout sayacı
                consec_empty = 0     # arka arkaya 0 ilan sayacı

                for page in range(1, total_pages + 1):
                    try:
                        driver.get(base.format(page=page))
                        WebDriverWait(driver, 12).until(
                            EC.presence_of_element_located(
                                (By.CSS_SELECTOR, "tr.listing-list-item")
                            )
                        )
                        time.sleep(1)
                        consec_timeouts = 0  # başarılı → sıfırla
                    except Exception:
                        consec_timeouts += 1
                        print(f"  S{page}:⏱({consec_timeouts}) ", end="", flush=True)

                        if consec_timeouts >= 3:
                            # 3 arka arkaya timeout → driver'ı yeniden başlat
                            print(f"\n  ⟳ Driver yenileniyor (blok algılandı)...", flush=True)
                            try:
                                driver.quit()
                            except Exception:
                                pass
                            time.sleep(random.uniform(8, 14))  # kısa mola
                            driver = get_driver()
                            consec_timeouts = 0
                            time.sleep(3)
                        else:
                            time.sleep(random.uniform(5, 9))
                        continue

                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    items = parse_listings(soup, marka)

                    saved = 0
                    for item in items:
                        if item["url"] and item["url"] not in existing_urls:
                            writer.writerow(item)
                            existing_urls.add(item["url"])
                            saved += 1

                    f.flush()
                    marka_saved += saved
                    total_saved += saved
                    print(f"  S{page}:{saved} ", end="", flush=True)

                    if saved == 0:
                        consec_empty += 1
                        if consec_empty >= 3:
                            print(f"\n  ✂ 3 boş sayfa → marka bitti, atlanıyor.", flush=True)
                            break
                    else:
                        consec_empty = 0

                    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

                print(f"\n  → {marka.upper()} toplam: {marka_saved} ilan | Genel toplam: {total_saved:,}")

                # Markayı tamamlandı olarak işaretle
                with open(PROGRESS_FILE, "a", encoding="utf-8") as pf:
                    pf.write(marka + "\n")
                done_markalar.add(marka)

                # Her marka sonrası driver yenile (temiz session)
                if m_idx < len(MARKALAR):
                    print(f"  ⟳ Sonraki marka için driver yenileniyor...", flush=True)
                    try:
                        driver.quit()
                    except Exception:
                        pass
                    time.sleep(random.uniform(4, 7))
                    driver = get_driver()

    except KeyboardInterrupt:
        print("\n\n⚠ Ctrl+C ile durduruldu. Kaydedilenler korundu.")
    finally:
        driver.quit()

    print("\n" + "=" * 55)
    print(f"✓ Tamamlandı! Toplam {total_saved:,} ilan")
    print(f"✓ Dosya: {OUTPUT_FILE}")
    print(f"Bitiş: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 55)

if __name__ == "__main__":
    main()
