import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# TMDb API Key (replace with your API key)
TMDB_API_KEY = 'your_tmdb_api_key'  # Replace with your TMDb API key

def read_data():
    data = pd.read_csv('./data/imdb_top_1000.csv')
    df = pd.DataFrame(data)
    return df

# Read and preprocess the data
df = read_data()
df = df.drop(columns=['Certificate', 'Gross', 'Meta_score'])

# Set up TF-IDF and Count Vectorizer for cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Overview'])
sim_vec = linear_kernel(tfidf_matrix, tfidf_matrix)
indexes = pd.Series(df.index, index=df['Series_Title'])

# Meta-data soup for improved similarity matching
df['meta_soup'] = df['Genre'] + ' ' + df['Director'] + ' ' + df['Star1'] + ' ' + df['Star2'] + ' ' + df['Star3'] + ' ' + df['Star4']
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['meta_soup'])
sim_vec = cosine_similarity(count_matrix, count_matrix)
indexes = pd.Series(df.index, index=df['Series_Title'])

# Function to fetch movie poster URL from csv
def fetch_movie_poster(title):
    movies = read_data()
    # column poster_link for movie with title
    poster_link = movies.loc[movies['Series_Title'] == title, 'Poster_Link'].values[0]
    return poster_link

# Function to get recommendations with poster URLs
def recommend_movies(title, sim_vec):
    title = title.strip()
    if title not in indexes:
        return {"error": "Movie title not found."}

    id_ = indexes[title]
    sim_movies = list(enumerate(sim_vec[id_]))
    sim_movies = sorted(sim_movies, key=lambda x: x[1], reverse=True)[1:11]  # Top 10 recommendations

    recommendations = []
    for i in sim_movies:
        movie_idx = i[0]
        movie_details = df.iloc[movie_idx][['Series_Title', 'Released_Year', 'IMDB_Rating', 'Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']]
        
        # Fetch poster URL for each movie
        poster_url = fetch_movie_poster(movie_details['Series_Title'])
        
        # Append movie details and poster URL to the recommendations list
        recommendations.append({
            "title": movie_details['Series_Title'],
            "year": movie_details['Released_Year'],
            "rating": movie_details['IMDB_Rating'],
            "genre": movie_details['Genre'],
            "director": movie_details['Director'],
            "stars": [movie_details['Star1'], movie_details['Star2'], movie_details['Star3'], movie_details['Star4']],
            "poster_url": poster_url
        })

    return recommendations

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    recommendations = recommend_movies(title, sim_vec)
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)
