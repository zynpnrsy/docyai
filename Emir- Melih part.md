# DocAI Feedback & Google Sheets Entegrasyonu (Emir & Melih)

Bu güncelleme ile DocAI projesine, kullanıcıların yapay zeka cevaplarını değerlendirebileceği bir **Geri Bildirim (Feedback) Sistemi** eklenmiştir. Veriler otomatik olarak bir Google Sheet dosyasına kaydedilir.

## 🚀 Neler Yapıldı?

1.  **Geri Bildirim Arayüzü:** Chat Assistant sekmesine 1-5 arası yıldız seçilebilen bir rating bölümü ve "Submit Rating" butonu eklendi.
2.  **Google Sheets Bağlantısı:** `feedback.py` dosyası oluşturularak verilerin (Soru, Cevap, Puan, Zaman) Google Sheets API üzerinden kaydedilmesi sağlandı.
3.  **Bağımlılıklar:** `gspread` ve `google-auth` kütüphaneleri projeye eklendi.

---

## 🛠️ Kurulum ve Kullanım Kılavuzu

Sistemin çalışması için aşağıdaki adımları takip etmelisiniz:

### 1. Kütüphaneleri Yükleyin

Terminalde şu komutu çalıştırın:

```bash
pip install gspread google-auth
```

### 2. Google Cloud Kurulumu (API Anahtarı)

1.  [Google Cloud Console](https://console.cloud.google.com/) adresine gidin.
2.  Yeni bir proje oluşturun.
3.  **APIs & Services > Library** kısmından **Google Sheets API** ve **Google Drive API**'yi bulup "Enable" diyerek aktifleştirin.
4.  **IAM & Admin > Service Accounts** kısmına gidin.
5.  "Create Service Account" diyerek bir hesap oluşturun.
6.  Oluşturduğunuz hesabın üzerine tıklayın, **Keys** sekmesine gelin ve **Add Key > Create New Key (JSON)** seçeneğini seçin.
7.  İnen JSON dosyasının ismini `credentials.json` olarak değiştirin ve projenin ana klasörüne kopyalayın.

### 3. Google Sheet Hazırlığı

1.  [Google Sheets](https://sheets.google.com/) üzerinden **"DocAI Feedback"** adında boş bir tablo oluşturun.
2.  `credentials.json` dosyasını açın ve içindeki `"client_email"` adresini kopyalayın.
3.  Google Sheet dosyanızın sağ üstündeki **Paylaş (Share)** butonuna tıklayın ve bu e-posta adresini **"Düzenleyici" (Editor)** olarak ekleyin.

### 4. .env Ayarları

Proje klasöründeki `.env` dosyasının şu şekilde olduğundan emin olun:

```env
GOOGLE_CREDENTIALS_FILE=credentials.json
GOOGLE_SHEET_NAME=DocAI Feedback
```

---

## 📊 Veri Formatı

Sistem her başarılı gönderimde Google Sheet'e şu sütunları ekler:

- **Question:** Kullanıcının sorduğu soru.
- **Answer:** Yapay zekanın verdiği cevap.
- **Rating:** Kullanıcının verdiği puan (1-5).
- **Timestamp:** İşlem saati.

Artık uygulamayı `python app.py` ile çalıştırıp test edebilirsiniz.
