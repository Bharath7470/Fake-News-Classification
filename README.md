#  Fake News Classification using Machine Learning

This internship project detects whether a news article is **Fake** or **Real** using machine learning and natural language processing (NLP). A trained Naive Bayes model is deployed through a Streamlit web app for real-time prediction.

## Project Details

- **Student Name:** Bharath  
- **Semester:** 5th Semester, B.Tech  
- **Project Title:** News Article Classification (Fake/Real)  
- **Internship Task:** Full ML pipeline + web interface

##  Tools and Technologies

- Python, Jupyter Notebook
- Pandas, Numpy, Scikit-learn, NLTK
- TF-IDF for feature extraction
- Naive Bayes classifier
- Joblib for saving model
- Streamlit for UI

##  Files in This Repository

| File | Description |
|------|-------------|
| `NewsClassification.ipynb` | Jupyter notebook for cleaning, training, and evaluating the model |
| `model.pkl` | Trained Naive Bayes model |
| `tfidf.pkl` | TF-IDF vectorizer |
| `app.py` | Streamlit app code for real-time prediction |
| `report.pdf` | Final project report |
| `README.md` | Project documentation (this file) |

##  Dataset Source

This project uses the following dataset from Kaggle:

🔗 [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

> ⚠️ **Note:**  
> The original CSV files (`Fake.csv`, `True.csv`) are **NOT included** in this repository due to GitHub's 25MB file size limit.  
> To run the notebook or app, download these files from the link above and place them in the project folder.

##  How to Run the Streamlit App

1. Install Streamlit (if not already installed):
   `pip install streamlit`
2. Run the app:
   `streamlit run app.py`
3. Paste any news article and click "Classify" to see whether it’s fake or real.

## Project Output

- Trained model classifies fake vs. real news with high accuracy
- Streamlit app allows easy, real-time text input and predictions
- Final report summarizes the full pipeline, results, and learning

## Author

**Bharath**  
5th Semester, B.Tech  
Fake News Detection Internship Project
