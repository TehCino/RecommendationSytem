# RecommendationSytem
Bachelor’s Thesis titled "Personalized Meal Recommendation System", showcasing the implementation and evaluation of different recommendation techniques (Collaborative Filtering, Content-Based, and Hybrid). Includes API deployment using FastAPI and performance benchmarking using Precision, Recall, and F1 Score.

## 📌 Project Overview

This repository contains a **hybrid meal recommendation system** that blends:
- **Collaborative Filtering (SVD)** using user-meal interaction data
- **Content-Based Filtering** using TF-IDF on meal category descriptions
- **Hybrid Approaches** (both weighted and non-weighted)

The FastAPI server hosts the hybrid (non-weighted) model for interactive recommendation via Swagger UI.

## 🔧 Technologies Used

| Tool            | Purpose                          |
|-----------------|----------------------------------|
| Python          | Core programming language        |
| Pandas, NumPy   | Data preprocessing and handling  |
| Scikit-learn    | ML algorithms (SVD, TF-IDF, NN)  |
| FastAPI         | API deployment                   |
| Jupyter Notebook| Model evaluation & experimentation |
