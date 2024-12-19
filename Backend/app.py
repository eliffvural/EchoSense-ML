from flask import Flask, jsonify, render_template, request
import librosa
import numpy as np
import joblib
import sounddevice as sd
import threading
import time
import noisereduce as nr
import whisper
import wave
import io
from flask_socketio import SocketIO, emit
import logging

app = Flask(__name__, template_folder='../Frontend')
socketio = SocketIO(app, cors_allowed_origins="*")

# Modelleri yükleme
speaker_model = joblib.load("../Model/random_forest_model.joblib")
scaler = joblib.load("../Model/scaler.joblib")
whisper_model = whisper.load_model("base")

# Mikrofon ve tahmin değişkenlerini düzenleme
is_recording = False
last_prediction_time = 0
prediction_interval = 1.5

# Konuşmacı etiketlerini belirleme
SPEAKER_LABELS = {0: "Elif", 1: "İrem", 2: "Nazlı"}

# Kategori anahtar kelimeleri
CATEGORY_KEYWORDS = {
    "Spor": [
        "futbol", "basketbol", "voleybol", "tenis", "kriket", "golf", "maraton", 
        "spor", "şampiyona", "fitness", "antrenman", "koşu", "maç", "hakem", 
        "puan", "gol", "turnuva", "antrenör", "stadyum", "rakip", "kulüp"
    ],
    
    "Teknoloji": [
        "bilgisayar", "telefon", "yapay zeka", "robot", "internet", "donanım", 
        "kodlama", "yazılım", "mobil", "blockchain", "kripto para", "drone", 
        "veri", "otomasyon", "5G", "IoT", "elektronik", "bulut bilişim", 
        "siber güvenlik", "bilişim", "big data", "robotik", "teknoloji"
    ],
    "Sanat": [
        "resim", "heykel", "fotoğrafçılık", "edebiyat", "şiir", "roman", 
        "hikaye", "film", "tiyatro", "bale", "opera", "müzik", "keman", 
        "piyano", "gitar", "davul", "performans", "sergi", "galeri", "sanatçı"
    ],
    "Ekonomi": [
        "borsa", "yatırım", "kripto para", "döviz", "fiyat", "enflasyon", 
        "tasarruf", "işsizlik", "maliye", "kredi", "gelir", "harcama", 
        "ekonomi", "faiz", "piyasa", "ticaret", "vergi", "bütçe", "finans", 
        "borsa endeksi", "iş dünyası", "kazanma"
    ],
    "Eğitim": [
        "okul", "öğretmen", "öğrenci", "ders", "üniversite", "kitap", "sınav", 
        "araştırma", "ödev", "çalışma", "müfredat", "kurs", "sertifika", 
        "matematik", "tarih", "fen bilimleri", "edebiyat", "akademi", "not", 
        "başarı", "öğretim", "eğitim sistemi"
    ],
    "Dünyadan Haberler": [
        "haber", "dünya", "politik", "savaş", "barış", "diplomasi", "seçim", 
        "protesto", "uluslararası", "anlaşma", "ekonomi", "lider", "gündem", 
        "baskı", "kriz", "doğal afet", "küresel", "mülteci", "terör", "birleşmiş milletler"
    ],
    "Tarih": [
        "imparatorluk", "antikkent", "medeniyet", "arkeoloji", "müze", "kazı", 
        "krallar", "hanedan", "savaş", "barış antlaşması", "zafer", "devrim", 
        "antik dönem", "osmanlı", "roma", "moğollar", "ortaçağ", "ilk çağ", 
        "yeni çağ", "çağdaş tarih", "destan", "yazıt", "kalıntı", "efsane"
    ],
    "Çocuklar": [
        "çocuk", "bebek", "oyun", "oyuncak", "eğitim", "masal", "hikaye", 
        "çizgi film", "aktivite", "park", "kreş", "anaokulu", "beslenme", 
        "çocuk şarkısı", "çizgi film karakteri", "aile"
    ],
    "Hava Durumu": [
        "hava", "yağmur", "güneşli", "fırtına", "kar", "bulutlu", "sis", 
        "soğuk", "sıcaklık", "rüzgar", "nem", "mevsim", "ilkbahar", "yaz", 
        "sonbahar", "kış", "hava tahmini", "meteoroloji", "iklim", "dolu", 
        "hortum"
    ],
    "Bilim": [
        "araştırma", "buluş", "keşif", "laboratuvar", "deney", "biyoloji", 
        "fizik", "kimya", "astronomi", "genetik", "tıp", "teknoloji", 
        "evrim", "mikroskop", "bilimsel çalışma", "uzay", "gezegen", 
        "nasa", "kuantum", "çevre bilimi", "enerji", "atom", "parçacık fiziği"
    ],
    "Oyun": [
        "oyun", "video oyunu", "playstation", "xbox", "bilgisayar oyunu", 
        "mobil oyun", "şampiyonluk", "oyuncu", "espor", "fps", "rpg", 
        "minecraft", "fortnite", "valorant", "league of legends", 
        "counter strike", "şifreleme", "simülasyon", "macera", "bulmaca"
    ],
    "Sosyal Hayat": [
        "arkadaş", "aile", "parti", "gezi", "tatil", "toplantı", "organizasyon", 
        "iletişim", "sosyal medya", "sinema", "alışveriş", "buluşma", "etkinlik", 
        "düğün", "yemek", "sosyal sorumluluk", "eğlence", "paylaşım"
    ],
    "Yemek": [
        "yemek", "tarif", "mutfak", "kahvaltı", "tatlı", "restoran", 
        "fast food", "salata", "çorba", "pizza", "makarna", "ızgara", 
        "vegan", "sebze", "meyve", "baharat", "lezzet", "menü", "şef", "dondurma"
    ],
    "Genel Sohbet": [
        "merhaba", "günaydın", "görüşürüz", "nasılsın", "bugün", "evet", 
        "hayır", "peki", "belki", "olabilir", "konu", "anlamadım", "şaka", 
        "güzel", "keyif", "zaman", "bazen", "aslında", "düşünce", "hayat"
    ],
    "Bilim ve Teknolojik İlerlemeler": [
        "yapay zeka", "robot", "uzay keşfi", "mars", "ay", "nasa", "rover", 
        "kuantum bilgisayar", "genetik mühendisliği", "crispr", "biyoteknoloji", 
        "nanoteknoloji", "elektrikli araç", "dronelar", "güneş enerjisi", 
        "hidrojen enerjisi", "uzay teleskobu", "grafen", "moleküler biyoloji", 
        "robotik cerrahi", "füzyon enerjisi", "yeni materyaller", 
        "büyük hadron çarpıştırıcısı", "parçacık fiziği", "yapay organlar", 
        "çevre bilimi", "iklim teknolojisi"
    ],
     "Çevre ve Doğa": [
        "iklim değişikliği", "çevre", "orman", "biyoçeşitlilik", "yenilenebilir enerji",
        "geri dönüşüm", "çevre koruma", "karbon salınımı", "doğal afet", "orman yangını",
        "sel", "doğa yürüyüşü", "hayvanlar", "deniz", "orman yaşamı", "sürdürülebilirlik",
        "çevre bilinci", "plastik atıklar"
    ],
    "Uzay Bilimi ve Astronomi": [
        "gezegenler", "yıldız", "galaksi", "uzay teleskobu", "kara delik", 
        "nötron yıldızı", "evrenin genişlemesi", "uzay istasyonu", 
        "mars görevleri", "ay keşfi", "exoplanet", "astronot", 
        "rover", "uzay aracı", "güneş sistemi", "komet", "asteroid", 
        "uzay yarışı", "supernova", "big bang"
    ],
    "Genetik ve Biyoteknoloji": [
        "dna", "genetik kod", "crispr", "gen terapisi", "genom düzenleme", 
        "biyoteknoloji", "hücre mühendisliği", "moleküler biyoloji", 
        "insan genom projesi", "yeni ilaçlar", "bakteri genetiği", 
        "genetik modifikasyon", "bitki genetiği", "klonlama", "yapay doku", 
        "hücre yenilenmesi", "hastalık genetiği", "gen aktarımı", 
        "genom analiz", "epigenetik"
    ],
     "Doğa Olayları": [
        "yağmur", "fırtına", "sel", "deprem", "volkan", "dolu", "tsunami",
        "kar", "çığ", "orman yangını", "gök gürültüsü", "kasırga", "hortum",
        "şimşek", "tornado", "kuraklık", "meteoroloji", "doğal afet", "iklim değişikliği"
    ],
    "Enerji ve Çevre Teknolojileri": [
        "güneş enerjisi", "hidrojen enerjisi", "rüzgar enerjisi", 
        "yenilenebilir enerji", "iklim değişikliği", "karbon salınımı", 
        "sıfır emisyon", "karbon yakalama", "sürdürülebilirlik", 
        "biyoyakıtlar", "nükleer enerji", "enerji verimliliği", 
        "çevre bilimi", "plastik geri dönüşüm", "orman koruma", 
        "su arıtma", "atık yönetimi", "iklim çözümleri", 
        "elektrikli araç", "akıllı şehir"
    ],
      "Memleket": [
        "köy", "kasaba", "şehir", "memleket", "sokak", "komşular", "köy kahvesi",
        "çiftçilik", "tarla", "bağ", "bahçe", "yerel halk", "anılar", 
        "memleket yemekleri", "mahalli", "folklor", "yerel müzik", "memleket havası", "bölge kültürü"
    ],
     "Ulaşım ve Seyahat": [
        "uçak", "tren", "otobüs", "tatil", "seyahat planları", "yolculuk", 
        "bilet", "otogar", "havalimanı", "turizm", "yurt dışı", "vize", "oteller", 
        "kamping", "doğa gezileri", "şehir turu", "seyahat rehberi", "tatil köyü"
    ]
}


# Özellik çıkarma
def extract_features_from_audio(audio_data, sr=16000):
    try:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=27)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        rms = librosa.feature.rms(y=audio_data)
        zcr = librosa.feature.zero_crossing_rate(y=audio_data)
        features = np.hstack((np.mean(mfcc, axis=1), np.mean(chroma, axis=1), np.mean(rms), np.mean(zcr)))
        return features
    except Exception as e:
        logging.error(f"Feature extraction error: {e}")
        return None

# Kategori tahmini
def predict_category(text):
    text = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(word in text for word in keywords):
            return category
    return "Kategori Bulunamadı"

# Mikrofon callback
def audio_callback(indata, frames, time_info, status):
    global last_prediction_time

    if status:
        print(f"Status: {status}")

    try:
        audio_data = indata[:, 0]  # Mono kanal
        print("1. Ses verisi alındı.")

        audio_data = nr.reduce_noise(y=audio_data, sr=16000)  # Gürültü azaltma
        print("2. Gürültü azaltma yapıldı.")

        features = extract_features_from_audio(audio_data)
        if features is not None:
            print(f"3. Özellikler çıkarıldı: {features}")  # Log
            current_time = time.time()

            if current_time - last_prediction_time >= prediction_interval:
                scaled_features = scaler.transform(features.reshape(1, -1))
                print(f"4. Özellikler ölçeklendi: {scaled_features}")  # Log

                prediction = model.predict(scaled_features)[0]
                print(f"5. Model Tahmini: {prediction}")  # Log

                speaker = SPEAKER_LABELS.get(prediction, "Bilinmiyor")
                print(f"6. Tahmin edilen konuşmacı: {speaker}")

                socketio.emit('speaker_update', {'speaker': speaker})
                last_prediction_time = current_time
        else:
            print("3.1 Özellik çıkarılamadı, None döndü.")
            socketio.emit('speaker_update', {'speaker': "Bilinmiyor"})

    except Exception as e:
        print(f"Audio işleme hatası: {e}")
        socketio.emit('speaker_update', {'speaker': "Bilinmiyor"})



# Ses kaydını başlat
def record_audio():
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
        while is_recording:
            time.sleep(0.5)

@socketio.on('start_recording')
def start_recording():
    global is_recording
    if not is_recording:
        is_recording = True
        threading.Thread(target=record_audio).start()
        print("Recording started...")

@socketio.on('stop_recording')
def stop_recording():
    global is_recording
    is_recording = False
    print("Recording stopped")
    emit('speaker_update', {'speaker': 'None'})
    emit('transcription_update', {'transcription': '', 'category': ''})

# Yeni: Analyze route (manuel metin gönderimi için)
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # 1. Gelen JSON verisini al
        data = request.json
        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "Boş metin gönderildi"}), 400
        print(f"Gelen metin: {text}")

        # 2. Kategori tahmini
        category = predict_category(text)
        print(f"Tahmin edilen kategori: {category}")

        # Yanıtı döndür
        return jsonify({
            "speaker": "Bilinmiyor",  # Bu route'da konuşmacı tahmini yapılmıyor
            "category": category
        })

    except Exception as e:
        print(f"Sunucu hatası: {e}")
        return jsonify({"error": "Sunucu hatası", "details": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    socketio.run(app, debug=True)
