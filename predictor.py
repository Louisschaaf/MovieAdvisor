import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from flask import Flask, request, jsonify
from ast import literal_eval

app = Flask(__name__)

# Initialize global variables for the model components
df2 = None
cosine_sim2 = None
indices = None

def read_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    return df

def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        return [i['name'] for i in x]
    return []

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    else:
        return ''

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title):
    global df2, cosine_sim2, indices

    # Normalize the input title for matching
    title = title.strip().lower()

    if title not in indices:
        return []
    
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim2[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices].tolist()

def setup_model():
    global df2, cosine_sim2, indices

    # Load data
    df1 = read_data('./data/tmdb_5000_credits.csv')
    df2 = read_data('./data/tmdb_5000_movies.csv')

    df1.columns = ['id', 'title', 'cast', 'crew']
    df2 = df2.merge(df1, on='id')

    # Rename the title column appropriately
    if 'title_x' in df2.columns and 'title_y' in df2.columns:
        df2['title'] = df2['title_x']  # Assuming title_x has the correct title data
        df2 = df2.drop(['title_x', 'title_y'], axis=1)  # Drop the redundant title columns
    elif 'title_x' in df2.columns:
        df2.rename(columns={'title_x': 'title'}, inplace=True)
    elif 'title_y' in df2.columns:
        df2.rename(columns={'title_y': 'title'}, inplace=True)

    # Calculate mean and quantile for weighted rating
    C = df2['vote_average'].mean()
    m = df2['vote_count'].quantile(0.9)

    # Filter movies for the recommendation model
    df2 = df2.copy().loc[df2['vote_count'] >= m]
    df2['score'] = df2.apply(weighted_rating, axis=1, m=m, C=C)
    df2 = df2.sort_values('score', ascending=False)

    # Process data for recommendation
    df2['overview'] = df2['overview'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df2['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df2[feature] = df2[feature].apply(literal_eval)

    df2['director'] = df2['crew'].apply(get_director)

    features = ['cast', 'keywords', 'genres']
    for feature in features:
        df2[feature] = df2[feature].apply(get_list)

    features = ['cast', 'keywords', 'director', 'genres']
    for feature in features:
        df2[feature] = df2[feature].apply(clean_data)

    df2['soup'] = df2.apply(create_soup, axis=1)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df2['soup'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    # Reset the index and create reverse mapping with normalized titles
    df2 = df2.reset_index()
    indices = pd.Series(df2.index, index=df2['title'].str.strip().str.lower()).drop_duplicates()


# Endpoint to get recommendations for a movie title
@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Please provide a movie title"}), 400
    
    recommendations = get_recommendations(title)
    if not recommendations:
        return jsonify({"error": f"No recommendations found for title: {title}"}), 404

    return jsonify(recommendations)

# Setup model when the server starts
setup_model()

if __name__ == '__main__':
    app.run(debug=True)
