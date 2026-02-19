<div align="right">
  <a href="README.md">🇹🇷 Türkçe</a> | <a href="README_EN.md">🇬🇧 English</a>
</div>

# 💎 <img src="https://flagcdn.com/w40/tr.png" width="32" alt="TR" style="vertical-align: middle;"> Elmas Fiyat Tahmini: Uçtan Uca ML Boru Hattı ve API

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=flat&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat&logo=pandas)

## 📌 Proje Özeti
Bu proje, elmasların fiyatını fiziksel özelliklerine (karat, kesim, renk, berraklık, derinlik, tabla ve boyutlar) dayanarak tahmin etmek için tasarlanmış uçtan uca eksiksiz bir makine öğrenmesi çözümüdür. 

Temel model eğitiminin ötesine geçerek; sağlam bir veri ön işleme boru hattı (pipeline) kurar, optimize edilmiş bir **Destek Vektör Regresyonu (SVR)** modeli uygular ve bu modeli **FastAPI** kullanarak gerçek zamanlı bir RESTful web servisi olarak dışa sunar (deploy).

## ✨ Temel Özellikler
* **🧹 Sağlam Veri Ön İşleme:** İmkansız fiziksel boyutların (sıfır değerleri) otomatik yönetimi ve Keşifçi Veri Analizi (EDA) odaklı aykırı değer (outlier) temizliği.
* **⚙️ Özellik Mühendisliği:** Kategorik değişkenler için `LabelEncoder` ve sayısal ölçeklendirme için `StandardScaler` entegrasyonu.
* **🧠 Makine Öğrenmesi:** Doğru fiyat tahmini için RBF çekirdekli (kernel), yüksek oranda ince ayar yapılmış bir SVR (Destek Vektör Regresyonu) kullanımı.
* **🚀 Gerçek Zamanlı API:** Tahminleri anında sunmak için FastAPI ile oluşturulmuş tam işlevli bir web arayüzü ve REST API.
* **📦 Model Serileştirme:** Sorunsuz dağıtım için `pickle` aracılığıyla güvenli bir şekilde kaydedilen uçtan uca boru hattı (Model + Ölçeklendirici + Kodlayıcılar).

## 🛠️ Teknoloji Yığını
* **Veri Bilimi:** `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Makine Öğrenmesi:** `scikit-learn` (SVR, Ön İşleme, Metrikler)
* **Web & API:** `FastAPI`, `Uvicorn`, `Jinja2`

## 📂 Proje Yapısı
```text
├── train_model.py         # Veri yükleme, temizleme ve model eğitim betiği
├── main.py                # FastAPI web sunucusu ve tahmin uç noktası (endpoint)
├── 10-diamonds.csv        # Ham veri seti
├── diamond_model_complete.pkl # Serileştirilmiş model ve ön işlemciler (Üretilen)
├── testdata.csv           # Değerlendirme için test veri seti (Üretilen)
├── requirements.txt       # Proje bağımlılıkları
└── templates/
    └── index.html         # API için önyüz (frontend) arayüzü
