from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

def lsa(X_tfidf):
    print("dimension reduction with lsa...")
    # Latent semantic analysis as proposed in
    # https://scikit-learn.org/1.5/auto_examples/text/plot_document_clustering.html
    # 1. Apply Singular Value Decomposition on tf-idf
    # 2. SVD is not normalized so do it to improve kMeans
    lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
    X_lsa = lsa.fit_transform(X_tfidf)
    return X_lsa, lsa

