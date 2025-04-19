
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# Sayfa ayarları
st.set_page_config(page_title="Restaurant Eşleştirici", layout="wide")

st.title("📊 Restaurant Eşleştirici")
st.markdown("Excel dosyasını yükleyin, sistem en iyi eşleşmeleri bulsun.")

MAX_MB = 10

# Dosya yükleyici
uploaded_file = st.file_uploader("Excel dosyasını yükle (.xlsx)", type=["xlsx"])

if uploaded_file:
    # Boyut kontrolü
    if uploaded_file.size > MAX_MB * 1024 * 1024:
        st.error(f"❌ Dosya çok büyük. Maksimum izin verilen boyut: {MAX_MB}MB")
        st.stop()

    try:
        df = pd.read_excel(uploaded_file)

        # Gerekli sütunlar
        required_cols = ['tygo_restaurant_name', 'restaurant_name', 'restaurant_code']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Excel dosyasında şu sütunlar eksik: {', '.join(required_cols)}")
            st.stop()

        # Veriyi hazırla
        df['restaurant_name'] = df['restaurant_name'].astype(str).fillna('')
        df['tygo_restaurant_name'] = df['tygo_restaurant_name'].astype(str).fillna('')

        with st.spinner("🔍 Eşleştiriliyor..."):
            tfidf = TfidfVectorizer()
            tfidf_ref = tfidf.fit_transform(df['restaurant_name'])
            tfidf_query = tfidf.transform(df['tygo_restaurant_name'])

            sim = cosine_similarity(tfidf_query, tfidf_ref)
            best_match = sim.argmax(axis=1)

            df['matched_restaurant_namety'] = df['restaurant_name'].iloc[best_match].values
            df['matched_restaurant_codety'] = df['restaurant_code'].iloc[best_match].values
            df['similarity_score'] = sim.max(axis=1)

        st.success("✅ Eşleştirme tamamlandı!")

        st.dataframe(df[['tygo_restaurant_name', 'matched_restaurant_namety', 'matched_restaurant_codety', 'similarity_score']].head(50))

        # Excel çıktısı
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Excel çıktısını indir",
            data=buffer,
            file_name="eslesme_sonuclari.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"⚠️ Hata oluştu: {str(e)}")

