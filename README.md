# Kanser Hastaligi Tahmini - Makine Ogrenmesi Projesi

## Proje Açiklamasi
Bu projede makine ogrenmesi algoritmalari kullanilarak meme kanseri tespiti yapilmaktadir.
Bir hastanin tumorunun iyi huylu mu kotu huylu mu oldugu tahmin edilmektedir.

## Kullanilan Veri Seti
- **Veri Seti:** Breast Cancer Wisconsin Dataset
- **Kaynak:** Scikit-learn kutuphanesi (sklearn.datasets.load_breast_cancer)
- **Link:** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- **Boyut:** 569 hasta, 30 ozellik
- **Etiketler:** 0 = Kotu Huylu (Kanser), 1 = Iyi Huylu (Saglikli)

## Veri On Isleme Adimlari
- Eksik veri kontrolu yapildi (eksik veri bulunmadi)
- Ozellikler StandardScaler ile olceklendirildi
- Veri %80 egitim %20 test olarak bolundu
- Korelasyon analizi yapildi

## Kullanilan Algoritmalar

### 1. Logistic Regression
Her ozelligin agirligini hesaplayarak sinif tahmini yapar. Temiz verilerde cok basarilidir.

### 2. Decision Tree
Ozelliklere gore dallanarak karar verir. Anlasilmasi kolay bir modeldir.

### 3. Random Forest
100 karar agacinin oylamasiyla tahmin uretir. Asiri ogrenmeye karsi dayaniklidir.

### 4. KNN (K-Nearest Neighbors)
Yeni hastaya en yakin 5 hastaya bakarak tahmin yapar. Komsuluk bazli bir algoritmadır.

## Model Performans Karsilastirmasi

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9825   | 0.9825    | 0.9825 | 0.9825   |
| Decision Tree       | 0.8860   | 0.8887    | 0.8860 | 0.8867   |
| Random Forest       | 0.9386   | 0.9390    | 0.9386 | 0.9387   |
| KNN                 | 0.9386   | 0.9390    | 0.9386 | 0.9387   |

## Sonuc ve Yorumlar
- En iyi model **Logistic Regression** oldu, %98.2 basari orani ile one cikti
- Decision Tree en dusuk performansi gosterdi, egitim verisini fazla ezberledi
- Veri seti temiz ve duzenli oldugu icin tum modeller yuksek basari gosterdi
- Gercek hastane verisinde basari oranlari daha dusuk cikabilir

## Kodlarin Nasil Calistirilacagi
1. Google Colab'i acin: https://colab.research.google.com
2. Untitled1.ipynb dosyasini yukleyin
3. Calisma zamani → Tumunu calistir secin
4. Hucreleri sirayla calistirin
