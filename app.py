from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
import pickle

app = Flask(__name__)

def parse_genres(x):
    if pd.isnull(x):
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return [genre.strip() for genre in x.split(',')]

def load_data():
    anime = pd.read_csv('data/anime-dataset-2023.csv')
    
    anime = anime[~anime['Genres'].isna()]  
    anime['Genres'] = anime['Genres'].apply(parse_genres)
   
    anime['Score'] = pd.to_numeric(anime['Score'], errors='coerce') 
    anime['Score'] = anime['Score'].fillna(0)
    
    anime['Synopsis'] = anime['Synopsis'].fillna('')
    anime['Type'] = anime['Type'].fillna('Unknown')
    
    return anime


def preprocess_data(anime):
    anime['genre_str'] = anime['Genres'].apply(lambda x: ' '.join(x))
       
    anime['content'] = anime['genre_str'] + ' ' + anime['Synopsis'] + ' ' + anime['Type']
    
    return anime

def create_model(anime):
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=5)
    tfidf_matrix = tfidf.fit_transform(anime['content'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

model_path = 'models/content_based_model.pkl'
if not os.path.exists(model_path):
    os.makedirs('models', exist_ok=True)
    print("Creating model...")
    anime = load_data()
    anime = preprocess_data(anime)
    cosine_sim = create_model(anime)
    with open(model_path, 'wb') as f:
        pickle.dump((anime, cosine_sim), f)
else:
    with open(model_path, 'rb') as f:
        anime, cosine_sim = pickle.load(f)

def get_recommendations(title, anime, cosine_sim, n=10):
    matches = anime[anime['Name'].str.lower() == title.lower()]
    if matches.empty:
        return pd.DataFrame()
    
    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    
    anime_indices = [i[0] for i in sim_scores]
    recommendations = anime.iloc[anime_indices].copy()
    recommendations['similarity'] = [i[1] for i in sim_scores]
    return recommendations

@app.route('/')
def index():
    popular_anime = anime.sort_values('Score', ascending=False).head(20)
    return render_template('index.html', top_anime=popular_anime)

@app.route('/recommend/<string:anime_title>')
def recommend(anime_title):
    matching_anime = anime[anime['Name'].str.lower() == anime_title.lower()]
    if matching_anime.empty:
        return redirect(url_for('index'))
    
    anime_details = matching_anime.iloc[0].to_dict()
    recommendations = get_recommendations(anime_title, anime, cosine_sim)
    return render_template('recommendations.html',
                         anime=anime_details,
                         recommendations=recommendations)

@app.route('/search')
def search():
    query = request.args.get('query', '').lower()
    if not query:
        return redirect(url_for('index'))
    
    results = anime[
        anime['Name'].str.lower().str.contains(query) |
        anime['Genres'].apply(lambda x: any(query in g.lower() for g in x)) |
        anime['Synopsis'].str.lower().str.contains(query)
    ]
    return render_template('index.html', search_results=results.head(20), query=query)

if __name__ == '__main__':
    app.run(debug=True)