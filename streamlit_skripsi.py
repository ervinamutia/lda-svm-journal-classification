import streamlit as st
import pandas as pd

import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

import joblib

# Read Dataset
df = pd.read_csv('dataTerlabel_withKey.csv', sep = ',', encoding = 'ISO-8859-1')


# PREPROCESSING DATA

# Case Folding
def casefolding(text):
    text = text.lower()
    text = re.sub(r"[-()\"#/@;:<>{}'+=%~|.!?,_]", " ", text)
    return text

# Tokenizing
def word_tokenize_wrapper(text):
    return word_tokenize(text)

# Stopwords Removal
stopwords_indo = set(stopwords.words('indonesian'))

def stopwordText(text):
    return[word for word in text if word not in stopwords_indo]

# Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

exception_words = {"bali"}

def stemmed_wrapper(term):
    if term in exception_words:
        return term
    return stemmer.stem(term)

term_dict = {}

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)

def stemmingText(document):
    return [term_dict.get(term, term) for term in document]




# LOAD MODEL
with open('TFIDFvectorizer.pkl', 'rb') as v:
    vectorizer = joblib.load(v)

with open('svm_model.pkl', 'rb') as m:
    model_svm = joblib.load(m)


# Function Klasifikasi Artikel
def klasifikasi(text):
    tfidf = vectorizer.transform([text])
    prediction = model_svm.predict(tfidf)
    if prediction == 0:
        category = "Aplikasi dan Evaluasi Sistem"
    elif prediction == 1:
        category = "Model dan Klasifikasi Data"
    elif prediction == 2:
        category = "Sistem dan Jaringan"
    elif prediction == 3:
        category = "Sistem Informasi dan Evaluasi Website"
    elif prediction == 4:
        category = "Analisis Data dan Pembelajaran"
    elif prediction == 5:
        category = "Pembelajaran dan Lingkungan"
    elif prediction == 6:
        category = "Citra dan Identifikasi"
    # elif prediction == 7:
    #     category = "Topik 8"
    # elif prediction == 8:
    #     category = "Topik 9"
    # elif prediction == 9:
    #     category = "Topik 10"
    else:
        category = "Error"
    return category

        

# STREAMLIT UI

st.title('Klasifikasi Artikel')
st.text('Silahkan inputkan data baru dengan mengisi judul artikel dan abstrak artikel!')

inputJudul = st.text_area("Judul Artikel")
inputAbstrak = st.text_area("Abstrak Artikel")

# button = st.button('Masukkan Data', type='primary')

if st.button('Masukkan Data', type='primary'):
    if inputAbstrak.strip() != " ":
        data = casefolding(inputAbstrak)
        data = word_tokenize_wrapper(data)
        data = stopwordText(data)
        data = stemmingText(data)
        data = " ".join(data)
        st.success(f'Prediksi: {klasifikasi(data)}')
    else:
        st.warning('ERROR NIH')