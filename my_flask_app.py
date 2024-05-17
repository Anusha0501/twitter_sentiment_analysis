from flask import Flask, render_template, request 
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')

a=joblib.load("finalized_svm_model.sav")

lemmatizer = WordNetLemmatizer()
app = Flask(__name__)

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens
custom_stopwords = [
    "said", "say", "says", "going", "like",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
]


# Custom TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words=custom_stopwords, tokenizer=tokenize_and_lemmatize)
df = pd.read_csv('Twitter_Data.csv')
file = pd.DataFrame(df)
file['clean_text'].fillna('', inplace=True)

# Custom TF-IDF Vectorization
# Custom TF-IDF Vectorization with reduced dimensions
tfidf_vectorizer = TfidfVectorizer(stop_words=custom_stopwords, tokenizer=tokenize_and_lemmatize, max_features=104321)


# Fit and transform the data
tfidf_matrix_custom = tfidf_vectorizer.fit_transform(file['clean_text'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def my_predict():
    if request.method == 'POST':
        text = request.form['text']
        text = ' '.join(tokenize_and_lemmatize(text))  # Join tokens into a single string
        text_vectorized = tfidf_vectorizer.transform([text])

        prediction = a.predict(text_vectorized)[0]
        if prediction == -1:
            category = 'Negative'
        elif prediction == 0:
            category = 'Neutral'
        else:
            category = 'Positive'
        return render_template('answer.html', prediction=category)


if __name__ == '__main__':
    app.run(debug=True)
