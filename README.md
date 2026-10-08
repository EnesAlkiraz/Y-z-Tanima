🤖 Biyometrik Yüz Tanıma Sistemi

Klasördeki referans fotoğraflarınızı kullanarak web kamerasından gerçek zamanlı yüz tanıma yapan ve herhangi bir fotoğrafı referanslarla karşılaştıran Python tabanlı bir yüz tanıma uygulaması.

Yüz özellikleri DeepFace kütüphanesi ve VGG-Face modeli ile çıkarılır, kosinüs mesafesi ile karşılaştırılır.

✨ Özellikler
📷 Canlı kamera tanıma: Web kamerasında gerçek zamanlı yüz tespiti ve doğrulama
🖼️ Fotoğraf testi: Seçtiğiniz bir fotoğrafı referanslarla karşılaştırır, işaretlenmiş sonuç görselini kaydeder
🔄 Otomatik veritabanı: Referans fotoğraflar bir kez taranır, embedding'ler referanslar.pkl dosyasına kaydedilir
⚡ Akıcı arayüz: Tanıma işlemi arka plan iş parçacığında (thread) çalışır, kamera görüntüsü takılmaz
🎨 Modern HUD: Köşe vurgulu çerçeve, yarı saydam bilgi bantları, FPS göstergesi
🇹🇷 Türkçe karakter desteği: TrueType font ile ğ, ü, ş, ı, ö, ç düzgün görüntülenir
🧭 Etkileşimli menü ve komut satırı argümanları
🧠 Nasıl Çalışır?
Proje klasöründeki referans fotoğraflardan VGG-Face embedding vektörleri çıkarılır ve birim vektöre normalize edilir.
Kamera modunda yüz, Haar Cascade ile hızlıca tespit edilir.
Tespit edilen yüz kırpılarak arka plandaki iş parçacığına gönderilir (yaklaşık her 0.25 saniyede bir).
Yüzün embedding'i, tüm referanslarla kosinüs mesafesi üzerinden karşılaştırılır.
En yakın mesafe eşik değerinin altındaysa kişi doğrulanır, değilse bilinmeyen olarak işaretlenir.
Durum	Renk	Anlamı
✅ DOĞRULANDI	Yeşil	Referanslarla eşleşti
❌ BİLİNMEYEN	Kırmızı	Eşleşme yok
🔍 ANALİZ EDİLİYOR	Sarı	Sonuç bekleniyor
📦 Kurulum
Gereksinimler
Python 3.9 – 3.11 (önerilir)
Web kamerası (canlı mod için)
Windows (font yolları ve CAP_DSHOW Windows için ayarlıdır, bkz. Notlar)
Adımlar
bash
# 1. Depoyu klonlayın
git clone https://github.com/KULLANICI_ADINIZ/REPO_ADI.git
cd REPO_ADI

# 2. (İsteğe bağlı) Sanal ortam oluşturun
python -m venv venv
venv\Scripts\activate

# 3. Bağımlılıkları yükleyin
pip install deepface opencv-python numpy pillow tf-keras

İlk çalıştırmada DeepFace, VGG-Face model ağırlıklarını otomatik olarak indirir (internet gerekir).

🚀 Kullanım
1. Referans fotoğraflarınızı ekleyin

Kendi fotoğraflarınızı (.jpg, .jpeg, .png, .webp) script ile aynı klasöre koyun. Farklı açı, ışık ve ifadelerde 5–10 net fotoğraf daha doğru sonuç verir.

test_, sonuc_ ve yakalanan_ ile başlayan dosyalar referans olarak sayılmaz.

2. Programı çalıştırın

Etkileşimli menü:

bash
python yuz_tanima.py

Komut satırı argümanları:

bash
python yuz_tanima.py --kamera              # Doğrudan canlı kamerayı başlat
python yuz_tanima.py --resim foto.jpg      # Tek bir fotoğrafı test et
python yuz_tanima.py --tara                # Referansları yeniden tara

Dosya adını (yuz_tanima.py) kendi script adınıza göre değiştirin.

Kamera modu kısayolları
Tuş	İşlev
Q / ESC	Çıkış
S	Ekran görüntüsü kaydet (yakalanan_*.jpg)
R	Referansları yeniden tara
⚙️ Yapılandırma

Script başındaki sabitleri değiştirerek davranışı ayarlayabilirsiniz:

Sabit	Varsayılan	Açıklama
MODEL_NAME	"VGG-Face"	Kullanılan yüz tanıma modeli
ESIK_DEGERI	0.65	Kosinüs mesafe eşiği. Düşürürseniz daha katı, yükseltirseniz daha esnek olur
KULLANICI_ADI	"ENES"	Tanındığında ekranda görünecek isim
VERITABANI_DOSYASI	"referanslar.pkl"	Embedding veritabanı dosyası
📁 Proje Yapısı
.
├── yuz_tanima.py        # Ana uygulama
├── referanslar.pkl      # Otomatik oluşur (embedding veritabanı)
├── foto1.jpg            # Referans fotoğraflarınız
├── foto2.jpg
└── sonuc_*.jpg          # Fotoğraf testi çıktıları
📝 Notlar
Windows odaklıdır: Font yolları C:/Windows/Fonts/ altındadır ve kamera cv2.CAP_DSHOW ile açılır. Linux/macOS'ta font yollarını değiştirmeniz gerekir. Font bulunamazsa varsayılan font kullanılır ve Türkçe karakterler bozuk görünebilir.
Referans fotoğrafları değiştirdiyseniz veritabanını yenilemek için menüden [3]'ü seçin, --tara kullanın ya da referanslar.pkl dosyasını silin.
Fotoğraf test modunda yalnızca fotoğraftaki ilk tespit edilen yüz değerlendirilir.
⚠️ Sınırlamalar ve Güvenlik Uyarısı
Bu proje eğitim ve kişisel kullanım amaçlıdır. Canlılık (liveness) tespiti içermez; bir fotoğraf veya ekran görüntüsü ile kandırılabilir. Kilit açma, kimlik doğrulama gibi güvenlik gerektiren yerlerde kullanmayın.
Kosinüs eşiği ve model doğruluğu ışık, açı ve kamera kalitesine göre değişir.
referanslar.pkl dosyası pickle formatındadır. Güvenmediğiniz kaynaklardan gelen .pkl dosyalarını yüklemeyin, çünkü pickle zararlı kod çalıştırabilir.
🔒 Gizlilik: Kendi yüz fotoğraflarınızı ve referanslar.pkl dosyasını GitHub'a yüklemeyin. Aşağıdaki .gitignore önerisini kullanın.
gitignore
*.jpg
*.jpeg
*.png
*.webp
*.pkl
venv/
__pycache__/
🛠️ Kullanılan Teknolojiler
DeepFace (VGG-Face)
OpenCV
NumPy
Pillow
TensorFlow / Keras
📄 Lisans

Bu proje MIT lisansı ile paylaşılmaktadır. (Lisans dosyasını eklemeyi unutmayın veya istediğiniz lisansla değiştirin.)

🤝 Katkıda Bulunma

Hata bildirimleri, öneriler ve pull request'ler memnuniyetle karşılanır. ⭐ Beğendiyseniz depoya yıldız vermeyi unutmayın!
