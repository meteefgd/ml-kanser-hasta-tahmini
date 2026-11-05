import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Meme Kanseri Tahmini", page_icon="🩺", layout="wide")

# Etiketler: 0 = Kötü huylu (kanser), 1 = İyi huylu
MODELLER = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}


@st.cache_resource
def hazirla():
    v = load_breast_cancer()
    X = pd.DataFrame(v.data, columns=v.feature_names)
    y = pd.Series(v.target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    egitilmis, satirlar = {}, []
    for ad, m in MODELLER.items():
        pipe = make_pipeline(StandardScaler(), m)
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        egitilmis[ad] = pipe
        satirlar.append({
            "Model": ad,
            "Accuracy": accuracy_score(yte, p),
            "Kanser Recall": recall_score(yte, p, pos_label=0),
            "Kanser Precision": precision_score(yte, p, pos_label=0),
            "F1 (kanser)": f1_score(yte, p, pos_label=0),
            "AUC-ROC": roc_auc_score(yte, proba),
            "5-Fold CV Acc": cross_val_score(make_pipeline(StandardScaler(), m), X, y, cv=cv).mean(),
        })
    cms = {ad: confusion_matrix(yte, egitilmis[ad].predict(Xte)) for ad in MODELLER}
    return X, y, Xte, yte, egitilmis, pd.DataFrame(satirlar), cms


X, y, Xte, yte, egitilmis, sonuc_df, cms = hazirla()

st.title("🩺 Meme Kanseri Tahmini")
st.caption("Eğitim amaçlı proje — tıbbi teşhis aracı DEĞİLDİR. Memenizde bir değişiklik fark ederseniz beklemeden bir doktora başvurun.")

tab4, tab1, tab2, tab3 = st.tabs(["🩺 Belirti Rehberi", "🔬 Model Tahmini", "📊 Model Karşılaştırma", "🗂 Veri Seti"])

# ---------------- TAHMİN ----------------
with tab1:
    st.info("🔬 **Meraklısına:** Bu ve sonraki sekmeler, projenin makine öğrenmesi tarafıdır. "
            "Modeller Breast Cancer Wisconsin veri setiyle eğitildi ve hücre çekirdeği ölçümlerini girdi olarak alır.")
    sol, sag = st.columns([1, 2])
    with sol:
        model_ad = st.selectbox("Model", list(MODELLER), index=0)
        idx = st.selectbox("Test setinden örnek hasta", range(len(Xte)),
                           format_func=lambda i: f"Hasta #{i}")
        if st.button("Bu hastanın değerlerini yükle"):
            for k, val in Xte.iloc[idx].items():
                st.session_state[k] = float(val)
            st.session_state["gercek"] = int(yte.iloc[idx])
        if st.button("Ortalamaya sıfırla"):
            for k in X.columns:
                st.session_state[k] = float(X[k].mean())
            st.session_state.pop("gercek", None)

    with st.expander("30 özelliği düzenle", expanded=False):
        cols = st.columns(3)
        girdi = []
        for i, k in enumerate(X.columns):
            lo, hi = float(X[k].min()), float(X[k].max())
            with cols[i % 3]:
                girdi.append(st.number_input(k, min_value=lo * 0.5, max_value=hi * 1.5,
                                             value=float(st.session_state.get(k, X[k].mean())),
                                             key=k, format="%.5f"))

    p1 = egitilmis[model_ad].predict_proba(pd.DataFrame([girdi], columns=X.columns))[0, 1]
    p0 = 1 - p1
    with sag:
        if p0 >= 0.5:
            st.error(f"### ❌ KÖTÜ HUYLU (kanser) — %{p0*100:.1f}")
        else:
            st.success(f"### ✅ İYİ HUYLU — %{p1*100:.1f}")
        st.progress(float(p0), text=f"Kötü huylu olasılığı: %{p0*100:.1f}")
        if "gercek" in st.session_state:
            g = "İyi huylu" if st.session_state["gercek"] == 1 else "Kötü huylu"
            st.info(f"Bu hastanın gerçek tanısı: **{g}**")
        st.caption("Not: Model, eğitim verisindeki dağılımın dışındaki değerlerde güvenilmez sonuç verir.")

# ---------------- KARŞILAŞTIRMA ----------------
with tab2:
    st.markdown("Tüm metrikler **gerçek model çıktılarından** hesaplanır "
                "(test seti: 114 hasta). **Kanser Recall** = gerçek kanserlerin kaçının yakalandığı; "
                "bu problemde en kritik metrik.")
    st.dataframe(sonuc_df.style.format({c: "{:.4f}" for c in sonuc_df.columns[1:]}),
                 width="stretch", hide_index=True)
    st.bar_chart(sonuc_df.set_index("Model")[["Accuracy", "Kanser Recall", "AUC-ROC", "5-Fold CV Acc"]])
    st.subheader("Confusion Matrix")
    c = st.columns(4)
    for col, (ad, cm) in zip(c, cms.items()):
        with col:
            st.markdown(f"**{ad}**")
            st.dataframe(pd.DataFrame(cm, index=["Gerçek: Kötü", "Gerçek: İyi"],
                                      columns=["Tahmin: Kötü", "Tahmin: İyi"]))

# ---------------- VERİ ----------------
with tab3:
    a, b, c3 = st.columns(3)
    a.metric("Hasta sayısı", len(X))
    b.metric("Kötü huylu", int((y == 0).sum()))
    c3.metric("İyi huylu", int((y == 1).sum()))
    st.dataframe(X.describe().T.round(3), width="stretch")

# ---------------- BELİRTİ REHBERİ ----------------
with tab4:
    st.subheader("Belirti Rehberi (kural tabanlı)")
    st.warning("Bu bölüm makine öğrenmesi modeli DEĞİL, basit bir bilgilendirme aracıdır. "
               "Teşhis koymaz; belirti seçmemiş olmanız kanser olmadığınız anlamına gelmez.")
    st.markdown("**Son zamanlarda fark ettiğiniz değişiklikleri işaretleyin:**")
    onemli = {
        "Memede veya koltuk altında yeni bir sertlik / ele gelen kitle": 2,
        "Meme ucundan kanlı veya kendiliğinden gelen akıntı": 2,
        "Meme derisinde çukurlaşma, portakal kabuğu görünümü veya kızarıklık": 2,
        "Memenin boyut ya da şeklinde belirgin, tek taraflı değişiklik": 2,
    }
    diger = {
        "Meme ucunda yeni içe çekilme": 1,
        "Meme ucu veya çevresinde pullanma / kabuklanma": 1,
        "Düzelmeyen, tek bölgede süren meme ağrısı": 1,
    }
    risk = {
        "Birinci derece akrabamda (anne, kız kardeş) meme kanseri öyküsü var": 1,
        "Daha önce meme ile ilgili biyopsi / tedavi öyküm var": 1,
    }
    puan, risk_var = 0, False
    for k, v in {**onemli, **diger}.items():
        if st.checkbox(k, key="b_" + k):
            puan = max(puan, v)
    st.markdown("**Risk faktörleri:**")
    for k in risk:
        if st.checkbox(k, key="r_" + k):
            risk_var = True

    st.divider()
    if puan == 2:
        st.error("### Bir doktora başvurmanız önerilir\n"
                 "Seçtiğiniz belirtiler değerlendirme gerektirir. Mümkün olan en kısa sürede "
                 "bir aile hekimine, genel cerrahi veya meme kliniğine başvurun. "
                 "Bu belirtiler çoğu zaman iyi huylu nedenlerle de olabilir; ayırt etmek doktorun işidir.")
    elif puan == 1 or risk_var:
        st.warning("### Doktorunuza danışın\n"
                   "Seçtiğiniz belirti veya risk faktörü bir sağlık uzmanıyla konuşmaya değer. "
                   "Yakın bir zamanda randevu alın ve düzenli kontrollerinizi aksatmayın.")
    else:
        st.info("### Belirti seçilmedi\n"
                "Bu, risk olmadığı anlamına gelmez. Yaşınıza uygun düzenli tarama ve kontrolleri "
                "sürdürün, memelerinizde bir değişiklik fark ederseniz beklemeden doktora gidin.")
