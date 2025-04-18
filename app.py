
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

st.title("Restoran Adı Eşleştirme AI Agent")

uploaded_file = st.file_uploader("Excel dosyasını yükleyin (tygo_restaurant_name, restaurant_name, restaurant_code sütunlarını içermeli)", type=["xlsx"])

if uploaded_file:
    try:
        data = pd.read_excel(uploaded_file)
        required_columns = ['tygo_restaurant_name', 'restaurant_name', 'restaurant_code']
        if not all(column in data.columns for column in required_columns):
            st.error("Excel dosyası gereken sütunları içermiyor: tygo_restaurant_name, restaurant_name, restaurant_code")
        else:
            # TF-IDF ve benzerlik hesaplama
            tfidf_vectorizer = TfidfVectorizer().fit(data['restaurant_name'].astype(str))
            namety_tfidf = tfidf_vectorizer.transform(data['restaurant_name'].astype(str))
            nameym_tfidf = tfidf_vectorizer.transform(data['tygo_restaurant_name'].astype(str))

            similarity_matrix = cosine_similarity(nameym_tfidf, namety_tfidf)
            best_match_indices = similarity_matrix.argmax(axis=1)

            data['matched_restaurant_namety'] = data['restaurant_name'].iloc[best_match_indices].values
            data['matched_restaurant_codety'] = data['restaurant_code'].iloc[best_match_indices].values
            data['similarity_score'] = similarity_matrix.max(axis=1)

            st.success("Eşleştirme tamamlandı. Aşağıdan sonucu ön izleyebilir ve indirebilirsiniz.")
            st.dataframe(data[['tygo_restaurant_name', 'matched_restaurant_namety', 'matched_restaurant_codety', 'similarity_score']])

            # Excel'e dönüştür
            output = io.BytesIO()
            data.to_excel(output, index=False)
            st.download_button("Sonuçları Excel Olarak İndir", data=output.getvalue(), file_name="eslesme_sonuclari.xlsx")

    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")

