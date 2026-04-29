# Kanser Hastalığı Tahmini — Makine Öğrenmesi Projesi

## Proje Açıklaması

Bu projede makine öğrenmesi algoritmaları kullanılarak meme kanseri tespiti yapılmaktadır.
Bir hastanın tümörünün iyi huylu mu yoksa kötü huylu mu olduğu tahmin edilmektedir.

---

## Kullanılan Veri Seti

- **Veri Seti:** Breast Cancer Wisconsin Dataset
- **Kaynak:** Scikit-learn kütüphanesi (`sklearn.datasets.load_breast_cancer`)
- **Link:** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- **Boyut:** 569 hasta, 30 özellik
- **Sınıf Dağılımı:** %62 iyi huylu (357), %38 kötü huylu (212) — **dengesiz veri**
- **Etiketler:** `0` = Kötü huylu (kanser), `1` = İyi huylu (sağlıklı)

---

## Veri Ön İşleme Adımları

- Eksik veri kontrolü yapıldı (eksik veri bulunmadı)
- Özellikler `StandardScaler` ile ölçeklendirildi
- Veri %80 eğitim / %20 test olacak şekilde bölündü
- Korelasyon analizi yapıldı
- ⚠️ **Eksik:** 5-fold cross-validation uygulanmadı — tek split sonuçları istatistiksel olarak güvenilmez

---

## Kullanılan Algoritmalar

### 1. Logistic Regression
Her özelliğin ağırlığını hesaplayarak sınıf tahmini yapar. Temiz, lineer ayrılabilir verilerde oldukça başarılıdır. Bu veri setinde en yüksek skoru vermiştir.

### 2. Decision Tree
Özelliklere göre dallanarak karar verir. Anlaşılması kolay ancak overfitting'e yatkın bir modeldir. Hyperparameter optimizasyonu yapılmadan bırakıldığından en düşük performansı göstermiştir.

### 3. Random Forest
100 karar ağacının oylamasıyla tahmin üretir. Aşırı öğrenmeye karşı dayanıklıdır ancak bu projede varsayılan parametrelerle kullanıldığından potansiyelinin altında kalmıştır.

### 4. KNN (K-Nearest Neighbors)
Yeni hastaya en yakın 5 hastaya bakarak tahmin yapar. Komşuluk temelli bu yöntem, özellik ölçeklendirmesine duyarlıdır.

---

## Model Performans Karşılaştırması

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9825   | 0.9825    | 0.9825 | 0.9825   |
| Decision Tree       | 0.8860   | 0.8887    | 0.8860 | 0.8867   |
| Random Forest       | 0.9386   | 0.9390    | 0.9386 | 0.9387   |
| KNN                 | 0.9386   | 0.9390    | 0.9386 | 0.9387   |

> ⚠️ **Önemli Not:** Metrikler tek bir train/test split'e dayanmaktadır. 5-fold cross-validation ile sonuçlar farklılık gösterebilir. Ayrıca sınıf dengesizliği nedeniyle sadece Accuracy değerine güvenmek yanıltıcı olabilir; AUC-ROC eğrisi daha güvenilir bir metrik olurdu.

---

## Bilinen Zayıf Noktalar ve İyileştirme Önerileri

| Sorun | Açıklama | Çözüm |
|-------|----------|-------|
| Tek train/test split | İstatistiksel güvenilirlik düşük | 5-fold cross-validation ekle |
| Hyperparameter optimizasyonu yok | Random Forest ve KNN varsayılan parametrelerle çalışıyor | GridSearchCV veya RandomizedSearchCV kullan |
| AUC-ROC eksik | Sınıf dengesizliğinde Accuracy yanıltıcı | ROC eğrisi ve Confusion Matrix ekle |
| Veri seti küçük | 569 örnek gerçek dünya için yetersiz | Daha büyük, gerçek hastane verisiyle test et |

---

## Sonuç ve Yorumlar

- En iyi model **Logistic Regression** oldu — %98.25 doğruluk oranıyla öne çıktı
- Logistic Regression'ın Random Forest'ı geçmesi, RF modelinin optimize edilmediğine işaret ediyor
- Decision Tree en düşük performansı gösterdi; eğitim verisini fazla ezberledi (overfitting)
- Veri seti temiz ve düzenli olduğu için tüm modeller görece yüksek başarı gösterdi
- Gerçek hastane verilerinde başarı oranları önemli ölçüde düşebilir

---

## Kodların Nasıl Çalıştırılacağı

1. Google Colab'i açın: https://colab.research.google.com
2. `Untitled1.ipynb` dosyasını yükleyin
3. **Çalışma Zamanı → Tümünü Çalıştır** seçin
4. Hücreleri sırasıyla çalıştırın

---

## Gereksinimler

```
scikit-learn
pandas
numpy
matplotlib
seaborn
```
