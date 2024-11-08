# This is a sample Python script.
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_dataset():
    categories = [
        "alt.atheism",
        "talk.religion.misc",
        "comp.graphics",
        "sci.space",
    ]

    dataset = fetch_20newsgroups(
        remove=("headers", "footers", "quotes"),
        subset="all",
        categories=categories,
        shuffle=True,
        random_state=42,
    )
    return dataset, len(categories)

def get_tfidf(data):

    vectorizer = TfidfVectorizer(
        stop_words="english",
    )
    X_tfidf = vectorizer.fit_transform(data)
    print(f"n_samples: {X_tfidf.shape[0]}, n_features: {X_tfidf.shape[1]}")
    return X_tfidf, vectorizer

def dimensionality_reduction(X_tfidf):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
    X_lsa = lsa.fit_transform(X_tfidf)
    explained_variance = lsa[0].explained_variance_ratio_.sum()
    return X_lsa, lsa

def get_top_terms(lsa, kmeans, vectorizer, true_k):
    original_space_centroids = lsa[0].inverse_transform(kmeans.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    for i in range(true_k):
        print(f"Cluster {i}: ", end="")
        for ind in order_centroids[i, :10]:
            print(f"{terms[ind]} ", end="")
        print()

def scatterplot(x,labels):
    print("scatterplot")
    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots()

    ax.scatter(x[:,0], x[:,1], c=labels, cmap='Paired')
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset, true_k = get_dataset()
    x_tfidf, vectorizer = get_tfidf(dataset.data)
    x_lsa, lsa = dimensionality_reduction(x_tfidf)


    # for token in tokens:
    #     print(token.text, token.pos_, token.dep_, token.lemma_, token.is_stop, token.is_alpha)
    kmeans = KMeans(
        n_clusters=true_k,
        max_iter=100,
        n_init=1,
    ).fit(x_lsa)
    get_top_terms(lsa, kmeans, vectorizer, true_k)

    distance_matrix = pairwise_distances(x_tfidf, x_tfidf, metric='cosine', n_jobs=-1)
    model = TSNE(metric="precomputed", init="random")
    Xpr = model.fit_transform(distance_matrix)
    scatterplot(Xpr,kmeans.labels_)

    tsne = TSNE(random_state=1, metric="cosine", init="random")
    embs = tsne.fit_transform(x_tfidf)
    scatterplot(embs,dataset.target)

    tsne = TSNE(random_state=1)
    embs = tsne.fit_transform(x_lsa)
    scatterplot(embs,kmeans.labels_)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
