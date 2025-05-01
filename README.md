# RecommendationSytem
Bachelor’s Thesis titled "Personalized Meal Recommendation System", showcasing the implementation and evaluation of different recommendation techniques (Collaborative Filtering, Content-Based, and Hybrid). Includes API deployment using FastAPI and performance benchmarking using Precision, Recall, and F1 Score.

## 📌 Project Overview

This recommendation system integrates:
- **Collaborative Filtering (SVD)** – Learns user preferences from historical meal interaction data.
- **Content-Based Filtering (TF-IDF + Nearest Neighbors)** – Suggests meals with similar categories or characteristics.
- **Hybrid Recommendation** – Combines both methods (non-weighted and weighted versions).

A real-time API was deployed using **FastAPI** and tested via **Swagger UI** in PyCharm.

User behaviour data (e.g., meal views and purchases) was tracked via **Google Analytics 4 (GA4)** and joined with metadata (e.g., meal categories) using **Google BigQuery**.

---

## 🔧 Technologies Used

| Tool / Platform            | Purpose                                                                 |
|----------------------------|-------------------------------------------------------------------------|
| **Python**                 | Core language for data processing and model development                |
| **Pandas, NumPy**          | Data manipulation and numerical operations                              |
| **Scikit-learn**           | Machine learning models (SVD, TF-IDF, Nearest Neighbors)                |
| **FastAPI**                | API deployment for serving real-time recommendations                    |
| **Jupyter Notebook**       | Model development and performance evaluation                            |
| **Google Analytics 4 (GA4)** | Captures user behaviour (e.g., meal views and purchases)             |
| **BigQuery**               | Links GA4 user behaviour data with meal metadata for model input        |
