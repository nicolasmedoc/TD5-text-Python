from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

def lsa(X_tfidf):
    print("dimension reduction with lsa...")
    lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
    X_lsa = lsa.fit_transform(X_tfidf)
    explained_variance = lsa[0].explained_variance_ratio_.sum()
    return X_lsa, lsa

