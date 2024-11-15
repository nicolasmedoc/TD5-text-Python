from sklearn.cluster import KMeans,AgglomerativeClustering

def kmeans(k, data):
    print("clustering kmeans...")
    return KMeans(
        n_clusters=k,
        max_iter=100,
        n_init=1,
    ).fit(data)

def agglomerative(k,data,linkage,affinity,distance_threshold=None):
    print("clustering agglomerative...")
    return AgglomerativeClustering(n_clusters=k,metric=affinity, linkage=linkage,distance_threshold=distance_threshold, compute_distances=True).fit(data)

