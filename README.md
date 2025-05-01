# RecommendationSytem
Bachelor’s Thesis titled "Personalized Meal Recommendation System", showcasing the implementation and evaluation of different recommendation techniques (Collaborative Filtering, Content-Based, and Hybrid). Includes API deployment using FastAPI and performance benchmarking using Precision, Recall, and F1 Score.

## 📌 Project Overview

This repository contains a **hybrid meal recommendation system** that blends:
- **Collaborative Filtering (SVD)** using user-meal interaction data
- **Content-Based Filtering** using TF-IDF on meal category descriptions
- **Hybrid Approaches** (both weighted and non-weighted)

The FastAPI server hosts the hybrid (non-weighted) model for interactive recommendation via Swagger UI.

User interaction data is captured via **Google Analytics 4 (GA4)** and linked to meal metadata using **BigQuery** before model training.

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
