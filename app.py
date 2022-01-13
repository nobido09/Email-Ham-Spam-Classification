import streamlit as st
import pickle
import nltk
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def cleanup_text(message):
    message = message.translate(str.maketrans('','',string.punctuation))
    words = [stemmer.stem(w) for w in message.split() if w.lower() not in stopwords.words('english') ]
    return ' '.join(words) 

def load_model(path='models\spam_classifier.pkl'):
    with open(path,'rb') as f:
        return pickle.load(f)

st.title('Email Spam detection')
with st.spinner('Loading our Spam Detection Model'):
    model = load_model()
    vectorizer = load_model('models/tfidf_vector.pkl')
    st.success('Model Loaded')

message = st.text_area('Enter the E-Mail',value='Hi there')
btn = st.button('Analyse')
if btn and len(message)> 10:
    stemmer = SnowballStemmer('english')
    clean_msg = cleanup_text(message)
    data = vectorizer.transform([clean_msg])
    data = data.toarray()
    prediction = model.predict(data)
    st.title('Prediction')
    if prediction[0] == 0:
        st.success("Normal Email")
    elif prediction[0] == 1:
        st.warning("Spam Email")
    else:
        st.error("Something is fishy")