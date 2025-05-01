import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from fastapi import FastAPI

app = FastAPI()

# Load dataset
file_path = r"" 
df = pd.read_csv(file_path)

# Convert data types
df['order_meal_id'] = df['order_meal_id'].astype(str)
df['user_id'] = df['user_id'].astype(str)
df['meal_categories'] = df['meal_categories'].fillna("")

# Ensure 'action' column is mapped correctly (2 for Purchase, 1 for View)
df['action'] = df['action'].apply(lambda x: 2 if x == 'Purchase' else 1)

# Merge meal categories for TF-IDF
df['meal_description'] = df['meal_categories']

# Train TF-IDF model on meal categories
tfidf = TfidfVectorizer(stop_words="english")
meal_matrix = tfidf.fit_transform(df['meal_description'])

# Train Nearest Neighbors model for meal similarity
nn_model = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
nn_model.fit(meal_matrix)


def get_similar_meals_by_category(item_id, top_n=5):
    """Finds meals with similar categories."""
    try:
        meal_idx = df[df['order_meal_id'] == str(item_id)].index[0]
        distances, indices = nn_model.kneighbors(meal_matrix[meal_idx], n_neighbors=top_n + 1)
        similar_meals = list(set(df.iloc[i]['order_meal_id'] for i in indices.flatten()[1:]))  # Exclude itself
        return similar_meals[:top_n]
    except IndexError:
        return []


# Create a user-meal interaction matrix
interaction_matrix = df.pivot_table(index='user_id', columns='order_meal_id', values='action', aggfunc='sum',
                                    fill_value=0)

# Convert to sparse matrix for efficiency
interaction_sparse = csr_matrix(interaction_matrix)

# Train SVD (Collaborative Filtering)
svd = TruncatedSVD(n_components=50)
user_factors = svd.fit_transform(interaction_sparse)
meal_factors = svd.components_


def recommend_for_user(user_id, top_n=5):
    """Recommend meals based on user interaction history."""
    try:
        user_idx = interaction_matrix.index.get_loc(str(user_id))
        scores = np.dot(user_factors[user_idx], meal_factors)
        meal_scores = list(zip(interaction_matrix.columns, scores))
        meal_scores = sorted(meal_scores, key=lambda x: x[1], reverse=True)[:top_n]
        return [meal[0] for meal in meal_scores]
    except IndexError:
        return []


def hybrid_recommend(user_id=None, item_id=None, top_n=5):
    """Blend Collaborative Filtering and Content-Based Filtering."""
    recommendations = set()

    if user_id:
        recommendations.update(recommend_for_user(user_id, top_n))

    if item_id:
        recommendations.update(get_similar_meals_by_category(item_id, top_n))

    return list(recommendations)[:top_n]


# FastAPI route to get recommendations
@app.get("/recommend/")
def get_recommendations(user_id: str = None, item_id: str = None, top_n: int = 5):
    """API to get recommendations based on user_id, item_id, or both."""
    recommendations = hybrid_recommend(user_id, item_id, top_n)
    return {"recommendations": recommendations}
