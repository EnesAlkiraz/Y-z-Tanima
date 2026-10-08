"""
====================================================================
               BİYOMETRİK YÜZ TANIMA SİSTEMİ
====================================================================
Bu script, klasördeki referans fotoğrafları kullanarak:
1. Web kamerasından gerçek zamanlı olarak yüzünüzü tanır.
2. Herhangi bir test fotoğrafını referanslarla karşılaştırır.
3. Yeni fotoğraflar eklendiğinde referans veritabanını günceller.
====================================================================
"""

import os
import sys
import time
import pickle
import argparse
import threading
from pathlib import Path

# Windows UTF-8 ve Anlık Satır Tamponu desteği
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

# TensorFlow gereksiz loglarını gizle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# DeepFace import
try:
    from deepface import DeepFace
except ImportError:
    print("[HATA] 'deepface' kütüphanesi bulunamadı. Lütfen 'pip install deepface' komutunu çalıştırın.")
    sys.exit(1)


# ====================================================================
# YAPILANDIRMA & SABİTLER
# ====================================================================
MODEL_NAME = "VGG-Face"
ESIK_DEGERI = 0.65           # Cosine mesafe eşiği (0.65 altı = eşleşti)
KULLANICI_ADI = "ENES"       # Tanındığında ekranda görünecek isim
VERITABANI_DOSYASI = "referanslar.pkl"
GECERLI_UZANTILAR = ('.jpg', '.jpeg', '.png', '.webp')

# Renk Paleti (BGR formatı OpenCV için, RGB formatı PIL için)
RENK_YESIL_BGR = (60, 220, 90)     # Doğrulandı (Neon Yeşil)
RENK_KIRMIZI_BGR = (50, 50, 230)    # Bilinmeyen (Kırmızı)
RENK_SARI_BGR = (0, 210, 255)      # Taranıyor (Sarı/Amber)
RENK_MAVI_BGR = (255, 180, 50)     # Bekleniyor (Mavi)

# TrueType Font yükleme (Türkçe karakter desteği için)
FONT_PATH_TITLE = "C:/Windows/Fonts/segoeuib.ttf" if os.path.exists("C:/Windows/Fonts/segoeuib.ttf") else "C:/Windows/Fonts/arialbd.ttf"
FONT_PATH_REGULAR = "C:/Windows/Fonts/segoeui.ttf" if os.path.exists("C:/Windows/Fonts/segoeui.ttf") else "C:/Windows/Fonts/arial.ttf"

try:
    FONT_LARGE = ImageFont.truetype(FONT_PATH_TITLE, 22)
    FONT_MEDIUM = ImageFont.truetype(FONT_PATH_TITLE, 17)
    FONT_SMALL = ImageFont.truetype(FONT_PATH_REGULAR, 14)
except Exception:
    FONT_LARGE = ImageFont.load_default()
    FONT_MEDIUM = ImageFont.load_default()
    FONT_SMALL = ImageFont.load_default()


# ====================================================================
# YÜZ TANIMA YÖNETİCİSİ (CORE ENGINE)
# ====================================================================
class YuzTanimaMotoru:
    def __init__(self, db_yolu=VERITABANI_DOSYASI):
        self.db_yolu = db_yolu
        self.referanslar = []
        self.yukle_veya_olustur()

    def yukle_veya_olustur(self, zorla_tara=False):
        """Veritabanı varsa yükler, yoksa veya zorla_tara=True ise oluşturur."""
        if os.path.exists(self.db_yolu) and not zorla_tara:
            try:
                with open(self.db_yolu, "rb") as f:
                    self.referanslar = pickle.load(f)
                if self.referanslar and len(self.referanslar) > 0:
                    print(f"[✓] {len(self.referanslar)} adet referans yüz veritabanından yüklendi.")
                    return
            except Exception as e:
                print(f"[!] Veritabanı okunamadı ({e}), yeniden taranıyor...")

        self.referanslari_tara()

    def referanslari_tara(self, klasor="."):
        """Klasördeki referans fotoğrafları tarayıp embedding çıkarır."""
        print("\n" + "=" * 55)
        print("    FOTOĞRAFLAR TARANIYOR VE VERİTABANI OLUŞTURULUYOR")
        print("=" * 55)

        dosyalar = [
            f for f in os.listdir(klasor)
            if f.lower().endswith(GECERLI_UZANTILAR)
            and not f.startswith(('test_', 'sonuc_', 'yakalanan_'))
        ]

        if not dosyalar:
            print("[!] Klasörde hiç referans fotoğraf bulunamadı!")
            return

        print(f"[*] Toplam {len(dosyalar)} referans fotoğraf bulundu. Yüzler çıkarılıyor...")
        yeni_referanslar = []

        for idx, dosya in enumerate(dosyalar, 1):
            dosya_yolu = os.path.join(klasor, dosya)
            print(f" [{idx}/{len(dosyalar)}] {dosya} işleniyor...", end="", flush=True)
            try:
                sonuclar = DeepFace.represent(
                    img_path=dosya_yolu,
                    model_name=MODEL_NAME,
                    detector_backend='opencv',
                    enforce_detection=False
                )
                if sonuclar and len(sonuclar) > 0:
                    emb = np.array(sonuclar[0]['embedding'], dtype=np.float32)
                    emb = emb / np.linalg.norm(emb)  # Birim vektör normalize
                    yeni_referanslar.append({
                        'dosya': dosya,
                        'embedding': emb
                    })
                    print(" -> [BAŞARILI]")
                else:
                    print(" -> [YÜZ BULUNAMADI]")
            except Exception as e:
                print(f" -> [HATA: {e}]")

        if yeni_referanslar:
            with open(self.db_yolu, "wb") as f:
                pickle.dump(yeni_referanslar, f)
            self.referanslar = yeni_referanslar
            print(f"\n[✓] {len(yeni_referanslar)} referans fotoğraf '{self.db_yolu}' dosyasına kaydedildi!\n")
        else:
            print("\n[!] Hiçbir fotoğraftan yüz embedding'i çıkarılamadı!\n")

    def karsilastir(self, yuz_emb):
        """Bir yüz embedding'ini referanslar ile karşılaştırır."""
        if not self.referanslar:
            return False, 1.0, 0.0, "Referans Yok"

        # Cosine distance = 1 - dot_product (normalize vektörler için)
        mesafeler = [
            (1.0 - float(np.dot(yuz_emb, ref['embedding'])), ref['dosya'])
            for ref in self.referanslar
        ]

        mesafeler.sort(key=lambda x: x[0])
        en_yakin_mesafe, en_iyi_dosya = mesafeler[0]

        eslesti = en_yakin_mesafe <= ESIK_DEGERI

        # İnsan dostu benzerlik yüzdesi hesabı
        if eslesti:
            # Mesafe 0 -> %100, Mesafe 0.65 -> %60
            benzerlik = max(0.0, min(100.0, (1.0 - (en_yakin_mesafe / ESIK_DEGERI)) * 40.0 + 60.0))
        else:
            benzerlik = max(0.0, (1.0 - en_yakin_mesafe) * 50.0)

        return eslesti, en_yakin_mesafe, benzerlik, en_iyi_dosya


# ====================================================================
# GÖRSEL ARAYÜZ YARDIMCILARI (HUD & ÇERÇEVELER)
# ====================================================================
def sik_kose_kutusu_ciz(img, x, y, w, h, renk, kalinlik=2, kose_uzunlugu=22):
    """Köşeleri vurgulu modern çerçeve çizer."""
    cv2.rectangle(img, (x, y), (x + w, y + h), renk, 1, cv2.LINE_AA)

    # Sol Üst
    cv2.line(img, (x, y), (x + kose_uzunlugu, y), renk, kalinlik, cv2.LINE_AA)
    cv2.line(img, (x, y), (x, y + kose_uzunlugu), renk, kalinlik, cv2.LINE_AA)
    # Sağ Üst
    cv2.line(img, (x + w, y), (x + w - kose_uzunlugu, y), renk, kalinlik, cv2.LINE_AA)
    cv2.line(img, (x + w, y), (x + w, y + kose_uzunlugu), renk, kalinlik, cv2.LINE_AA)
    # Sol Alt
    cv2.line(img, (x, y + h), (x + kose_uzunlugu, y + h), renk, kalinlik, cv2.LINE_AA)
    cv2.line(img, (x, y + h), (x, y + h - kose_uzunlugu), renk, kalinlik, cv2.LINE_AA)
    # Sağ Alt
    cv2.line(img, (x + w, y + h), (x + w - kose_uzunlugu, y + h), renk, kalinlik, cv2.LINE_AA)
    cv2.line(img, (x + w, y + h), (x + w, y + h - kose_uzunlugu), renk, kalinlik, cv2.LINE_AA)


def yari_saydam_serit(img, y1, y2, renk=(20, 20, 25), opaklik=0.65):
    """Görüntünün üst veya altına şık yarı saydam bant ekler."""
    h, w, _ = img.shape
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), renk, -1)
    cv2.addWeighted(overlay, opaklik, img, 1 - opaklik, 0, img)


def turkce_metin_ciz(cv_img, metinler):
    """
    OpenCV görüntüsü üzerine PIL kullanarak Türkçe TrueType yazı yazar.
    metinler: [(x, y, metin, font, (r, g, b)), ...]
    """
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for x, y, metin, font, renk_rgb in metinler:
        draw.text((x, y), metin, font=font, fill=renk_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ====================================================================
# MOD 1: CANLI KAMERA İLE YÜZ TANIMA
# ====================================================================
def kamera_ile_tani(motor):
    """Web kamerasından gerçek zamanlı, akıcı (60 FPS) yüz tanıma."""
    print("\n[+] Web kamerası başlatılıyor...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[HATA] Web kamerası açılamadı! Lütfen kameranızın bağlı ve erişilebilir olduğundan emin olun.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Haar Cascade hızlı yüz tespiti
    cascade_yolu = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_yolu)

    # Threading paylaşım değişkenleri
    kilit = threading.Lock()
    son_crop = None
    yeni_crop_var = threading.Event()
    durdur = False

    tanima_sonucu = {
        'durum': 'BEKLENIYOR',      # 'DOGRULANDI', 'BILINMEYEN', 'TARIYOR', 'BEKLENIYOR'
        'eslesti': False,
        'mesafe': 1.0,
        'benzerlik': 0.0,
        'dosya': '',
        'guncelleme_zamani': 0
    }

    def arka_plan_tanima_isparcacigi():
        nonlocal durdur, son_crop
        while not durdur:
            if not yeni_crop_var.wait(timeout=0.1):
                continue
            yeni_crop_var.clear()

            with kilit:
                if son_crop is None:
                    continue
                islem_crop = son_crop.copy()

            try:
                # DeepFace ile yüz crop'unu doğrudan VGG-Face'e besle
                rep = DeepFace.represent(
                    img_path=islem_crop,
                    model_name=MODEL_NAME,
                    detector_backend='skip',
                    enforce_detection=False
                )
                if rep and len(rep) > 0:
                    emb = np.array(rep[0]['embedding'], dtype=np.float32)
                    emb = emb / np.linalg.norm(emb)
                    eslesti, mesafe, benzerlik, dosya = motor.karsilastir(emb)

                    with kilit:
                        tanima_sonucu['durum'] = 'DOGRULANDI' if eslesti else 'BILINMEYEN'
                        tanima_sonucu['eslesti'] = eslesti
                        tanima_sonucu['mesafe'] = mesafe
                        tanima_sonucu['benzerlik'] = benzerlik
                        tanima_sonucu['dosya'] = dosya
                        tanima_sonucu['guncelleme_zamani'] = time.time()
            except Exception:
                pass

    # Arka plan iş parçacığını başlat
    worker = threading.Thread(target=arka_plan_tanima_isparcacigi, daemon=True)
    worker.start()

    fps_zaman = time.time()
    fps_sayac = 0
    fps_degeri = 30
    son_islem_zamani = 0

    print("[✓] Kamera aktif! Çıkmak için penceredeyken 'Q' veya 'ESC' tuşuna basın.")
    pencere_adi = "Biyometrik Yuz Tanima - Canli Kamera"
    cv2.namedWindow(pencere_adi, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Aynalama (doğal selfie görünümü)
            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape

            # FPS hesabı
            fps_sayac += 1
            if time.time() - fps_zaman >= 1.0:
                fps_degeri = fps_sayac
                fps_sayac = 0
                fps_zaman = time.time()

            # Hızlı yüz tespiti (Haar Cascade)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yuzler = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(80, 80)
            )

            suanki_zaman = time.time()
            metin_listesi = []

            if len(yuzler) > 0:
                # En büyük yüzü al (ana kullanıcı)
                ana_yuz = max(yuzler, key=lambda b: b[2] * b[3])
                x, y, w, h = ana_yuz

                # Her 0.25 saniyede bir yeni crop'u arka plana gönder
                if suanki_zaman - son_islem_zamani > 0.25:
                    son_islem_zamani = suanki_zaman
                    margin_x = int(w * 0.1)
                    margin_y = int(h * 0.1)
                    y1 = max(0, y - margin_y)
                    y2 = min(H, y + h + margin_y)
                    x1 = max(0, x - margin_x)
                    x2 = min(W, x + w + margin_x)
                    with kilit:
                        son_crop = frame[y1:y2, x1:x2].copy()
                        yeni_crop_var.set()

                # Güncel tanıma sonucunu oku
                with kilit:
                    durum = tanima_sonucu['durum']
                    benzerlik = tanima_sonucu['benzerlik']
                    mesafe = tanima_sonucu['mesafe']
                    son_guncelleme = tanima_sonucu['guncelleme_zamani']

                # Sonuç 2 saniyeden eskiyse taranıyor durumuna geç
                if suanki_zaman - son_guncelleme > 2.0:
                    durum = 'TARIYOR'

                # Çerçeve ve etiket renkleri
                if durum == 'DOGRULANDI':
                    kutu_renk = RENK_YESIL_BGR
                    etiket = f"DOĞRULANDI: {KULLANICI_ADI}  (%{benzerlik:.0f})"
                    alt_etiket = f"Benzerlik Mesafesi: {mesafe:.2f}"
                    badge_renk_rgb = (40, 200, 70)
                elif durum == 'BILINMEYEN':
                    kutu_renk = RENK_KIRMIZI_BGR
                    etiket = "BİLİNMEYEN KİŞİ"
                    alt_etiket = "Kayıtlı fotoğraflarla eşleşmedi"
                    badge_renk_rgb = (230, 40, 40)
                else:
                    kutu_renk = RENK_SARI_BGR
                    etiket = "YÜZ ANALİZ EDİLİYOR..."
                    alt_etiket = "Lütfen kameraya bakın"
                    badge_renk_rgb = (255, 190, 20)

                # Şık köşe kutusu çiz
                sik_kose_kutusu_ciz(frame, x, y, w, h, kutu_renk, kalinlik=3, kose_uzunlugu=25)

                # Yüz altı bilgi kartı
                kart_y = min(H - 45, y + h + 8)
                cv2.rectangle(frame, (x, kart_y), (x + max(w, 220), kart_y + 40), (25, 25, 30), -1)
                cv2.rectangle(frame, (x, kart_y), (x + max(w, 220), kart_y + 40), kutu_renk, 1)

                metin_listesi.append((x + 6, kart_y + 4, etiket, FONT_MEDIUM, badge_renk_rgb))
                metin_listesi.append((x + 6, kart_y + 22, alt_etiket, FONT_SMALL, (200, 200, 200)))

            else:
                # Yüz bulunamadı
                with kilit:
                    tanima_sonucu['durum'] = 'BEKLENIYOR'

            # Üst ve Alt HUD Bantları
            yari_saydam_serit(frame, 0, 42, renk=(20, 20, 25), opaklik=0.75)
            yari_saydam_serit(frame, H - 32, H, renk=(20, 20, 25), opaklik=0.75)

            # Üst Bar Metinleri
            durum_metni = "● YÜZ TESPİT EDİLDİ" if len(yuzler) > 0 else "○ YÜZ BEKLENİYOR"
            durum_renk = (40, 220, 80) if len(yuzler) > 0 else (200, 200, 200)
            metin_listesi.append((12, 10, "BİYOMETRİK YÜZ TANIMA SİSTEMİ", FONT_LARGE, (255, 255, 255)))
            metin_listesi.append((W - 200, 12, durum_metni, FONT_SMALL, durum_renk))
            metin_listesi.append((W - 80, 12, f"{fps_degeri} FPS", FONT_SMALL, (180, 180, 180)))

            # Alt Bar Metinleri
            rehber = f"[Q/ESC] Çıkış   |   [S] Ekran Görüntüsü Kaydet   |   Referans: {len(motor.referanslar)} Fotoğraf"
            metin_listesi.append((12, H - 25, rehber, FONT_SMALL, (200, 200, 200)))

            # Türkçe metinleri kareye çiz
            frame = turkce_metin_ciz(frame, metin_listesi)

            # Ekranda göster
            cv2.imshow(pencere_adi, frame)

            tus = cv2.waitKey(1) & 0xFF
            if tus in (ord('q'), ord('Q'), 27):  # Q veya ESC
                break
            elif tus in (ord('s'), ord('S')):
                kayit_adi = f"yakalanan_{int(time.time())}.jpg"
                cv2.imwrite(kayit_adi, frame)
                print(f"[✓] Ekran görüntüsü '{kayit_adi}' olarak kaydedildi.")
            elif tus in (ord('r'), ord('R')):
                print("[*] Referanslar arka planda yenileniyor...")
                motor.referanslari_tara()

    finally:
        durdur = True
        yeni_crop_var.set()
        cap.release()
        cv2.destroyAllWindows()
        print("[✓] Kamera kapatıldı.")


# ====================================================================
# MOD 2: FOTOĞRAF DOSYASI İLE TEST ETME
# ====================================================================
def resim_test_et(motor, resim_yolu):
    """Belirtilen bir resim dosyasındaki yüzü referanslarla karşılaştırır."""
    if not os.path.exists(resim_yolu):
        print(f"[HATA] '{resim_yolu}' dosyası bulunamadı!")
        return

    print("\n" + "=" * 55)
    print(f"    FOTOĞRAF TESTİ: {os.path.basename(resim_yolu)}")
    print("=" * 55)

    img = cv2.imread(resim_yolu)
    if img is None:
        print("[HATA] Fotoğraf dosyası okunamadı!")
        return

    H, W, _ = img.shape
    print("[*] Fotoğraftaki yüz analiz ediliyor...")

    try:
        sonuclar = DeepFace.represent(
            img_path=resim_yolu,
            model_name=MODEL_NAME,
            detector_backend='opencv',
            enforce_detection=False
        )
    except Exception as e:
        print(f"[HATA] Yüz analizi yapılamadı: {e}")
        return

    if not sonuclar:
        print("[!] Fotoğrafta yüz tespit edilemedi.")
        return

    # İlk yüzü al
    ilk = sonuclar[0]
    facial_area = ilk.get('facial_area', {})
    x = facial_area.get('x', 0)
    y = facial_area.get('y', 0)
    w = facial_area.get('w', W)
    h = facial_area.get('h', H)

    emb = np.array(ilk['embedding'], dtype=np.float32)
    emb = emb / np.linalg.norm(emb)

    eslesti, mesafe, benzerlik, en_iyi_dosya = motor.karsilastir(emb)

    print("-" * 55)
    print(f" Tespit Edilen Yüz Konumu : x={x}, y={y}, w={w}, h={h}")
    print(f" En Yakın Referans        : {en_iyi_dosya}")
    print(f" Cosine Mesafe            : {mesafe:.4f} (Eşik Değeri: {ESIK_DEGERI})")
    print(f" Benzerlik Skoru          : %{benzerlik:.1f}")
    print("-" * 55)

    if eslesti:
        print(f" [SONUÇ]  DOĞRULANDI! Bu fotoğraftaki kişi SİZSİNİZ ({KULLANICI_ADI}).")
        kutu_renk = RENK_YESIL_BGR
        baslik = f"DOĞRULANDI: {KULLANICI_ADI}  (%{benzerlik:.0f})"
        renk_rgb = (40, 220, 80)
    else:
        print(" [SONUÇ] ❌ EŞLEŞMEDİ! Bu fotoğraftaki kişi referanslarınızla eşleşmiyor.")
        kutu_renk = RENK_KIRMIZI_BGR
        baslik = f"BİLİNMEYEN KİŞİ  (Mesafe: {mesafe:.2f})"
        renk_rgb = (230, 40, 40)

    print("=" * 55 + "\n")

    # Sonuç görselini hazırla
    gorsel = img.copy()
    sik_kose_kutusu_ciz(gorsel, x, y, w, h, kutu_renk, kalinlik=3, kose_uzunlugu=35)

    # Üst banner
    yari_saydam_serit(gorsel, 0, 50, renk=(20, 20, 25), opaklik=0.8)
    metinler = [
        (15, 12, baslik, FONT_LARGE, renk_rgb),
        (15, H - 30 if H > 60 else 10, f"En İyi Eşleşme: {en_iyi_dosya} | Mesafe: {mesafe:.4f}", FONT_SMALL, (220, 220, 220))
    ]
    yari_saydam_serit(gorsel, H - 40, H, renk=(20, 20, 25), opaklik=0.8)

    gorsel = turkce_metin_ciz(gorsel, metinler)

    # Sonuç dosyasını kaydet
    cikti_adi = f"sonuc_{Path(resim_yolu).stem}.jpg"
    cv2.imwrite(cikti_adi, gorsel)
    print(f"[✓] İşaretlenmiş sonuç görseli '{cikti_adi}' olarak kaydedildi.")

    # Ekranda göster (isteğe bağlı, pencere boyutu ayarlanabilir)
    max_h = 800
    if H > max_h:
        olcek = max_h / H
        gorsel_kucuk = cv2.resize(gorsel, (int(W * olcek), max_h))
    else:
        gorsel_kucuk = gorsel

    cv2.imshow("Test Sonucu (Kapatmak icin herhangi bir tusa basin)", gorsel_kucuk)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ====================================================================
# MOD 3: DOSYA SEÇİCİ İLE TEST ETME (TKINTER)
# ====================================================================
def dosya_sec_ve_test_et(motor):
    """Windows dosya seçme penceresi açarak test eder."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        dosya = filedialog.askopenfilename(
            title="Test Edilecek Fotoğrafı Seçin",
            filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png *.webp"), ("Tüm Dosyalar", "*.*")]
        )
        root.destroy()
        if dosya:
            resim_test_et(motor, dosya)
        else:
            print("[!] Dosya seçimi iptal edildi.")
    except Exception as e:
        print(f"[!] Dosya seçici açılamadı ({e}), lütfen dosya adını elle girin.")
        dosya_yolu = input("Test edilecek resmin yolu: ").strip('\"\' ')
        if dosya_yolu:
            resim_test_et(motor, dosya_yolu)


# ====================================================================
# ANA PROGRAM & ETKİLEŞİMLİ MENÜ
# ====================================================================
def ana_menu():
    parser = argparse.ArgumentParser(description="Biyometrik Yüz Tanıma Sistemi")
    parser.add_argument("--kamera", action="store_true", help="Doğrudan canlı kamerayı başlat")
    parser.add_argument("--resim", type=str, help="Belirtilen resmi test et")
    parser.add_argument("--tara", action="store_true", help="Referans fotoğrafları yeniden tara")
    args = parser.parse_args()

    # Motoru başlat
    motor = YuzTanimaMotoru()

    if args.tara:
        motor.referanslari_tara()
        return

    if args.kamera:
        kamera_ile_tani(motor)
        return

    if args.resim:
        resim_test_et(motor, args.resim)
        return

    # Argümansız çalıştırıldığında etkileşimli menüyü göster
    while True:
        print("\n" + "=" * 55)
        print("         🤖 BİYOMETRİK YÜZ TANIMA SİSTEMİ")
        print("=" * 55)
        print(f" Referans Veritabanı : {len(motor.referanslar)} Fotoğraf Yüklü")
        print(f" Tanınacak Kişi      : {KULLANICI_ADI}")
        print(f" Yapay Zeka Modeli   : {MODEL_NAME}")
        print("-" * 55)
        print(" [1] 📷 Canlı Kamera (Webcam) ile Beni Tanı")
        print(" [2] 🖼️  Fotoğraf Seçerek Test Et")
        print(" [3] 🔄 Referans Fotoğrafları Yeniden Tara")
        print(" [4] ❌ Çıkış")
        print("=" * 55)

        secim = input("Seçiminiz (1-4): ").strip()

        if secim == "1":
            kamera_ile_tani(motor)
        elif secim == "2":
            dosya_sec_ve_test_et(motor)
        elif secim == "3":
            motor.referanslari_tara()
        elif secim in ("4", "q", "exit", "cikis"):
            print("[✓] Programdan çıkılıyor. Görüşmek üzere!")
            break
        else:
            print("[!] Geçersiz seçim, lütfen 1, 2, 3 veya 4 yazın.")


if __name__ == "__main__":
    ana_menu()
