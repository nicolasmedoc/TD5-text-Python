# This is a sample Python script.
import dataset
import textprocessing
import clustering
import dimred
import scatterplot
import projection
from sklearn.metrics import pairwise_distances


def get_cluster_top_terms(lsa, vectorizer, kmeans, k):
    print("getting cluster_top_terms")
    original_space_centroids = lsa[0].inverse_transform(kmeans.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    cluster_top_terms = []
    for i in range(k):
        top_terms = []
        print(f"Cluster {i}: ", end="")
        for ind in order_centroids[i, :10]:
            top_terms.append(terms[ind])
            print(f"{terms[ind]} ", end="")
        cluster_top_terms.append(top_terms)
        print()
    return cluster_top_terms

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset, true_k = dataset.get20newsgroups()
    x_tfidf, vectorizer = textprocessing.get_tfidf(dataset.data)
    x_lsa, lsa = dimred.lsa(x_tfidf)

    kmeans = clustering.kmeans(true_k, x_lsa)
    topterms = get_cluster_top_terms(lsa, vectorizer, kmeans, true_k)

    # distance_matrix = pairwise_distances(x_tfidf, x_tfidf, metric='cosine', n_jobs=-1)
    # model = TSNE(metric="precomputed", init="random")
    # Xpr = model.fit_transform(distance_matrix)
    # scatterplot.show(Xpr,kmeans.labels_)

    proj_euclidean = projection.tsne_euclidean(x_lsa)
    proj_cosine = projection.tsne_cosine(x_tfidf)

    scatterplot.show(proj_euclidean,kmeans.labels_)
    scatterplot.show(proj_cosine,dataset.target)
