import numpy as np
from flask import Flask, request, jsonify
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import re

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Preprocesamiento de texto
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub("[^A-Za-z1-9 ]", "", text)
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def preprocess_text_for_tfidf(text):
    return ' '.join(preprocess_text(text))

# Función para recomendar productos
def get_similar_products(product_id, products):
    # Buscar el producto inicial
    product = next((p for p in products if p['id'] == product_id), None)
    if not product:
        return None

    # Combinar la descripción con la categoría
    combined_texts_for_tfidf = [
        f"{p['category']} {p['description']}" for p in products
    ]
    combined_texts_for_w2v = [
        preprocess_text(f"{p['category']} {p['description']}") for p in products
    ]

    # TF-IDF y Similitud de Coseno
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts_for_tfidf)
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Modelo Word2Vec
    w2v_model = Word2Vec(sentences=combined_texts_for_w2v, vector_size=100, window=5, min_count=1, workers=4)

    def word2vec_similarity(desc1, desc2):
        desc1_vec = np.sum([w2v_model.wv[word] for word in desc1 if word in w2v_model.wv], axis=0)
        desc2_vec = np.sum([w2v_model.wv[word] for word in desc2 if word in w2v_model.wv], axis=0)
        return cosine_similarity([desc1_vec], [desc2_vec])[0][0]

    # Obtener índice del producto
    index = next(i for i, p in enumerate(products) if p['id'] == product_id)

    # Calcular similitudes combinadas
    combined_similarities = []
    for i in range(len(products)):
        if i != index:
            sim_tfidf = cosine_sim[index][i]
            sim_w2v = word2vec_similarity(combined_texts_for_w2v[index], combined_texts_for_w2v[i])
            combined_score = (sim_tfidf + sim_w2v) / 2
            combined_similarities.append((i, combined_score))
    
    # Ordenar productos por similitud
    combined_similarities = sorted(combined_similarities, key=lambda x: x[1], reverse=True)

    # Seleccionar las 5 mejores recomendaciones
    recommended_products = [products[i[0]] for i in combined_similarities[:5]]
    
    return recommended_products

@app.route('/recommend/price/<int:product_id>', methods=['GET'])
def recommend_price(product_id):
    response = requests.get('https://fakestoreapi.com/products')
    products = response.json()

    recommended_products = get_similar_products(product_id, products)

    if recommended_products is None:
        return jsonify({'error': 'Producto no encontrado'}), 404
    
    # Recomendar precios
    recommended_prices = [{ "title": p['title'], "price": p['price'], "description": p['description'] } for p in recommended_products]
    
    return jsonify(recommended_prices)

@app.route('/recommend/category/<int:product_id>', methods=['GET'])
def recommend_category(product_id):
    response = requests.get('https://fakestoreapi.com/products')
    products = response.json()

    recommended_products = get_similar_products(product_id, products)

    if recommended_products is None:
        return jsonify({'error': 'Producto no encontrado'}), 404
    
    # Recomendar categorías
    recommended_categories = [{ "title": p['title'], "category": p['category'], "description": p['description'] } for p in recommended_products]
    
    return jsonify(recommended_categories)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
