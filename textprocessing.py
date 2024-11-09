from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf(data):
    print("vectorizing with tf-idf...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
    )
    X_tfidf = vectorizer.fit_transform(data)
    print(f"n_samples: {X_tfidf.shape[0]}, n_features: {X_tfidf.shape[1]}")
    return X_tfidf, vectorizer

