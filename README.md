# Amazon Review Sentiment Analysis

An end-to-end Natural Language Processing (NLP) project that classifies Amazon product reviews as positive or negative using TF-IDF and Logistic Regression.

## Project Overview
This project downloads the Amazon Reviews dataset dynamically using `kagglehub`, preprocesses the text data (cleaning, lowercasing, stopword removal), converts text into numerical features using TF-IDF, and trains a Logistic Regression classifier to predict sentiment.

## Dataset
* **Source:** [Kaggle - Amazon Reviews (`bittlingmayer/amazonreviews`)](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
* **Labels:** 
  * `__label__1`: Negative review
  * `__label__2`: Positive review

## Pipeline Steps
1. **Data Loading:** Fetches the `.bz2` compressed text files using `kagglehub` and streams data into a Pandas DataFrame.
2. **Text Preprocessing:** Converts text to lowercase, removes numbers, punctuation, and English stopwords.
3. **Feature Extraction:** Transforms text into numerical vectors using `TfidfVectorizer`.
4. **Model Training & Evaluation:** Splits data into train/test sets, trains a `LogisticRegression` model, and evaluates performance using accuracy, precision, recall, and F1-score.

## Technologies Used
* Python
* Pandas, NumPy
* Scikit-Learn
* NLTK
* Kagglehub

## How to Run
1. Clone this repository or open the notebook in Kaggle/Jupyter.
2. Ensure you have the required libraries installed:
   ```bash
   pip install kagglehub pandas scikit-learn nltk
