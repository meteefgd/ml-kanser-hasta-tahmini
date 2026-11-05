[README.md](https://github.com/user-attachments/files/34421123/README.md)
# Meme Kanseri Tahmini — Makine Öğrenmesi + Web Arayüzü

Breast Cancer Wisconsin veri seti ile bir tümörün **iyi huylu** mu **kötü huylu** mu olduğunu tahmin eden 4 modelin karşılaştırması ve bu modelleri kullanan Streamlit arayüzü.

> Eğitim amaçlı projedir, tıbbi teşhis aracı değildir.

## Veri Seti
- Kaynak: `sklearn.datasets.load_breast_cancer`
- 569 hasta, 30 özellik, eksik veri yok
- Etiketler: `0` = kötü huylu (212), `1` = iyi huylu (357) — hafif dengesiz
- %80 eğitim / %20 test (`stratify=y`, `random_state=42`), `StandardScaler` Pipeline içinde

## Sonuçlar (gerçek model çıktıları, test seti = 114 hasta)

| Model | Accuracy | Kanser Recall | Kanser Precision | F1 (kanser) | AUC-ROC | 5-Fold CV Acc |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9762 | 0.9762 | 0.9762 | 0.9954 | 0.9737 |
| Decision Tree | 0.9123 | 0.9286 | 0.8478 | 0.8864 | 0.9157 | 0.9104 |
| Random Forest | 0.9561 | 0.9286 | 0.9512 | 0.9398 | 0.9939 | 0.9561 |
| KNN (k=5) | 0.9561 | 0.9286 | 0.9512 | 0.9398 | 0.9788 | 0.9631 |

**Yorum**
- Kanser tespitinde en kritik metrik *kanser recall*'dır (gerçek kanserlerin kaçı yakalandı). Bir kanseri kaçırmak yanlış alarmdan çok daha pahalıdır.
- Logistic Regression her metrikte önde, ancak test setinde 114 hasta olduğu için 1-2 hastalık farklar gürültü sayılır. 5-fold CV'de LR, RF ve KNN birbirine yakındır (%95.6 – %97.4).
- Decision Tree her ölçüte göre en zayıf modeldir (budanmamış tek ağaç, aşırı öğrenmeye yatkın).
- Veri seti temiz ve küçüktür; gerçek hastane verisinde skorların düşmesi beklenir.
- Hiperparametre optimizasyonu (GridSearchCV) yapılmadı; sonraki adım olarak eklenebilir.

## Dosyalar
- `kanser_tahmini.ipynb` — analiz ve model karşılaştırması
- `app.py` — Streamlit arayüzü (tahmin, model karşılaştırma, veri seti sekmeleri)
- `requirements.txt`

## Çalıştırma
```
pip install -r requirements.txt
streamlit run app.py
```

## Yayınlama (Streamlit Community Cloud)
1. Dosyaları GitHub reposuna yükle.
2. https://share.streamlit.io adresinde GitHub ile giriş yap, **New app** de.
3. Repo ve `app.py` dosyasını seç, **Deploy**'a bas.
4. Oluşan `*.streamlit.app` linkini paylaş.
