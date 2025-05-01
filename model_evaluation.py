import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Load and preprocess
file_path = r""
df = pd.read_csv(file_path)

df['order_meal_id'] = df['order_meal_id'].astype(str)
df['user_id'] = df['user_id'].astype(str)
df['meal_categories'] = df['meal_categories'].fillna("")
df['action'] = df['action'].apply(lambda x: 2 if x == 'Purchase' else 1)
df['meal_description'] = df['meal_categories']

# Train and test data
def train_test_split_by_user(df, test_size=0.2):
    train_data, test_data = [], []
    for user_id, group in df.groupby('user_id'):
        if len(group) < 2:
            train_data.append(group)
        else:
            train, test = train_test_split(group, test_size=test_size, random_state=42)
            train_data.append(train)
            test_data.append(test)
    return pd.concat(train_data), pd.concat(test_data)

df_train, df_test = train_test_split_by_user(df)

# Content-Based Filtering
tfidf = TfidfVectorizer(stop_words="english")
meal_matrix = tfidf.fit_transform(df['meal_description'])
nn_model = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
nn_model.fit(meal_matrix)

def get_similar_meals_by_category(item_id, top_n=5):
    try:
        meal_idx = df[df['order_meal_id'] == str(item_id)].index[0]
        distances, indices = nn_model.kneighbors(meal_matrix[meal_idx], n_neighbors=top_n + 1)
        similar_meals = list(set(df.iloc[i]['order_meal_id'] for i in indices.flatten()[1:]))
        return similar_meals[:top_n]
    except IndexError:
        return []

# Collaborative Filtering + SVD 
interaction_matrix = df_train.pivot_table(index='user_id', columns='order_meal_id', values='action', aggfunc='sum', fill_value=0)
interaction_sparse = csr_matrix(interaction_matrix)
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(interaction_sparse)
meal_factors = svd.components_

def recommend_for_user(user_id, top_n=5):
    try:
        user_idx = interaction_matrix.index.get_loc(str(user_id))
        scores = np.dot(user_factors[user_idx], meal_factors)
        meal_scores = list(zip(interaction_matrix.columns, scores))
        meal_scores = sorted(meal_scores, key=lambda x: x[1], reverse=True)[:top_n]
        return [meal[0] for meal in meal_scores]
    except:
        return []

# Hybrid Recommendation
def hybrid_recommend(user_id=None, item_id=None, top_n=5):
    recs = set()
    if user_id:
        recs.update(recommend_for_user(user_id, top_n))
    if item_id:
        recs.update(get_similar_meals_by_category(item_id, top_n))
    return list(recs)[:top_n]

def hybrid_weighted_recommend(user_id=None, item_id=None, top_n=5, alpha=0.7):
    collab_scores = {}
    content_scores = {}

    if user_id:
        try:
            user_idx = interaction_matrix.index.get_loc(str(user_id))
            collab_raw_scores = np.dot(user_factors[user_idx], meal_factors)
            for meal_id, score in zip(interaction_matrix.columns, collab_raw_scores):
                collab_scores[meal_id] = score
        except KeyError:
            pass

    if item_id:
        try:
            meal_idx = df[df['order_meal_id'] == str(item_id)].index[0]
            distances, indices = nn_model.kneighbors(meal_matrix[meal_idx], n_neighbors=top_n + 10)
            for i, dist in zip(indices.flatten()[1:], distances.flatten()[1:]):
                meal_id = df.iloc[i]['order_meal_id']
                content_scores[meal_id] = 1 - dist
        except IndexError:
            pass

    combined_scores = {}
    meal_ids = set(collab_scores.keys()).union(content_scores.keys())
    for meal_id in meal_ids:
        c_score = collab_scores.get(meal_id, 0)
        t_score = content_scores.get(meal_id, 0)
        combined_scores[meal_id] = alpha * c_score + (1 - alpha) * t_score

    sorted_recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [meal[0] for meal in sorted_recommendations[:top_n]]

# test first 100 users
def evaluate_model_subset(method_func, df_test, method_name="Collaborative", top_n=5, num_users=100):
    precision_list = []
    recall_list = []
    f1_list = []

    test_truth = defaultdict(set)
    for _, row in df_test.iterrows():
        if row['action'] >= 1:  # Includes both views and purchases
            test_truth[row['user_id']].add(row['order_meal_id'])

    sampled_users = list(test_truth.keys())[:num_users]

    for user_id in sampled_users:
        item_id = df_test[df_test['user_id'] == user_id]['order_meal_id'].values[0]
        if method_name == "Content":
            recommended = method_func(item_id, top_n=top_n)
        elif method_name == "Hybrid":
            recommended = hybrid_recommend(user_id=user_id, item_id=item_id, top_n=top_n)
        elif "Weighted" in method_name:
            recommended = method_func(user_id=user_id, item_id=item_id, top_n=top_n)
        else:
            recommended = method_func(user_id, top_n=top_n)

        relevant = test_truth.get(user_id, set())
        if not relevant:
            continue

        true_positive = len(set(recommended) & relevant)
        precision = true_positive / len(recommended) if recommended else 0
        recall = true_positive / len(relevant) if relevant else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "Method": method_name,
        "Precision@5": np.mean(precision_list),
        "Recall@5": np.mean(recall_list),
        "F1 Score@5": np.mean(f1_list)
    }

# Run Evaluation
collab_metrics_100 = evaluate_model_subset(recommend_for_user, df_test, method_name="Collaborative", top_n=5)
content_metrics_100 = evaluate_model_subset(get_similar_meals_by_category, df_test, method_name="Content", top_n=5)
hybrid_metrics_100 = evaluate_model_subset(hybrid_recommend, df_test, method_name="Hybrid", top_n=5)
hybrid_weighted_metrics_100 = evaluate_model_subset(
    method_func=lambda user_id, item_id=None, top_n=5: hybrid_weighted_recommend(user_id=user_id, item_id=item_id, top_n=top_n, alpha=0.7),
    df_test=df_test,
    method_name="Hybrid (Weighted)",
    top_n=5
)


results_df_100 = pd.DataFrame([
    collab_metrics_100,
    content_metrics_100,
    hybrid_metrics_100,
    hybrid_weighted_metrics_100
])

print(results_df_100)
