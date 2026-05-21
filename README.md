# Sensör Füzyonu ve Lokalizasyon Kullanarak LiDAR Tabanlı Otonom Navigasyon

[cite_start]Bu proje, bir mobil robotun 2B ortamda LiDAR, IMU ve tekerlek enkoderi verilerini kullanarak sensör füzyonu (Kalman Filtresi) gerçekleştirmesini, lokalizasyon yapmasını ve hedefe güvenli bir şekilde ulaşmasını hedefleyen bir otonom navigasyon çalışmasıdır[cite: 13, 58].

## 📋 Proje Gereksinimleri
[cite_start]Proje, aşağıdaki teknik standartlara göre geliştirilmiştir[cite: 58]:
- [cite_start]**Ortam:** 2B yerleşim alanı, en az 10 engel, başlangıç ve hedef noktaları[cite: 35, 36].
- [cite_start]**Sensör Füzyonu:** LiDAR, IMU ve tekerlek enkoderi verilerinin birleştirilerek Genişletilmiş Kalman Filtresi (EKF) ile işlenmesi zorunludur[cite: 58, 62].
- [cite_start]**Robot Modeli:** Non-holonomic (araba tipi) mobil robot kinematik modeli kullanılmıştır[cite: 73, 76].
- [cite_start]**Navigasyon:** Engellerden kaçınma, hedefe ulaşma ve dinamik yeniden planlama yetenekleri[cite: 72].

## 🚀 Kurulum ve Kullanım
Proje Python ortamında geliştirilmiştir. Çalıştırmak için:

1. **Gereksinimleri Yükle:**
   ```bash
   pip install -r requirements.txt
   
   python main.py
   
Sonuçlar ve Görsel Çıktılar Proje kapsamında üretilen görsel çıktılar ve analizler results/ dizininde bulunmaktadır:  
Ortam Haritası: Engelleri içeren 2B yerleşim planı.  
Robot Yol Planı: Planlanan rota ve izlenen gerçek yolun karşılaştırması.  
Lokalizasyon: Sensör füzyonu ile tahmin edilen konumun gerçek yol ile karşılaştırması.  
Hata Analizi: MAE (Ortalama Mutlak Hata) hesaplamaları ve hata grafikleri. 

Yapay Zeka Kullanım Beyanı
Bu projenin geliştirilmesi sürecinde yapay zeka araçlarından yardımcı kaynak olarak faydalanılmıştır.  
Kullanılan Araçlar: Gemini Flash PRO 3.1.  
Yapay Zekanın Kullanıldığı Bölümler:
Kalman Filtresi kodunun ilk taslağını oluşturma.  
Kod hatalarının ayıklanması ve hata ayıklama süreçlerine destek.  
README ve rapor metninin dil düzenlemeleri.  
Öğrencinin Katkıları:Proje senaryosunu ve sistem mimarisini tasarlama.  
Kodları test etme, çalıştırma ve gerekli düzeltmeleri yapma.  
Sonuç grafikleri, hata analizi ve değerlendirme yorumlarını hazırlama. 

Kaynakça
Bu çalışmada IEEE formatı temel alınmıştır.  
V. Ušinskis, M. Nowicki, A. Dzedzickis ve V. Bučinskas, "Sensor-fusion based navigation for autonomous mobile robot," Sensors, 2025.  
Y. Ou, Y. Cai, Y. Sun ve T. Qin, "Autonomous navigation by mobile robot with sensor fusion based on deep reinforcement learning," Sensors, 2024.  
B. Zhang ve C. Li, "The optimization and application research of the RRT-APF-based path planning algorithm," Electronics, 2024.  