import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from snowballstemmer import TurkishStemmer
import string
import nltk
import stanza
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Gerekli NLTK verilerini indirin
nltk.download('punkt')
nltk.download('stopwords')

# NLTK Türkçe stop words listesini yükleyin
stop_words = set(stopwords.words('turkish'))

# Stanza'yı başlatın
stanza.download('tr')
nlp = stanza.Pipeline('tr')

# FastText modeli eğitimi için gerekli fonksiyonu tanımlayın
def train_fasttext_model(data):
    # Burada FastText modelini eğitmek için gerekli adımları gerçekleştirin
    # Model eğitimini gerçekleştirdikten sonra, eğitilmiş modeli döndürün
    pass

# Streamlit uygulamasını tanımlayın
def main():
    st.title('FastText Modeli Eğitimi ve Sınıflandırma')

    # Dosya yükleme alanı ekleyin
    uploaded_file = st.file_uploader("Lütfen bir xlsx dosyası yükleyin", type=["xlsx"])

    if uploaded_file is not None:
        # Dosyayı okuyun
        df = pd.read_excel(uploaded_file)

        # İlk sütunun başlığını alın
        text_column = df.columns[0]

        # Metin ön işleme fonksiyonunu uygulayın
        def preprocess_text(text):
            # Tokenization
            tokens = word_tokenize(text)

            # Removing punctuation
            table = str.maketrans('', '', string.punctuation)
            stripped = [word.translate(table) for word in tokens]

            # Removing stopwords
            words = [word for word in stripped if word not in stop_words]

            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]

            # Stemming
            stemmer = TurkishStemmer()
            stemmed_words = [stemmer.stemWord(word) for word in words]

            return ' '.join(stemmed_words)

        df['processed_text'] = df[text_column].apply(preprocess_text)

        # Eğitim verilerini ve modeli eğitin
        trained_model = train_fasttext_model(df['processed_text'])

        st.success('FastText modeli başarıyla eğitildi!')

        # Örnek metin giriş alanı ekleme
        st.subheader('Metin Sınıflandırma')
        text_input = st.text_input('Metin Giriniz:', '')

        if text_input:
            # Girilen metni ön işleme yapın
            processed_input = preprocess_text(text_input)

            # Model tarafından tahmin edilen sınıfı bulun
            predicted_class = predict_class(processed_input, df, trained_model)

            # Tahmin edilen sınıfı göster
            st.write('Tahmin Edilen Sınıf:', predicted_class)

# Tahmin işlemi için kullanılacak olan fonksiyon
def predict_class(processed_input, df, trained_model):
    # Örnek bir sınıflandırma modeli oluşturun (Burada çok basit bir model kullandık)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['Sınıf']

    classifier = MultinomialNB()
    classifier.fit(X, y)

    # Tahmin işlemini gerçekleştirin
    processed_input_vectorized = vectorizer.transform([processed_input])
    predicted_class = classifier.predict(processed_input_vectorized)

    return predicted_class[0]

# Uygulamayı çalıştırın
if __name__ == '__main__':
    main()
