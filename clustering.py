from sklearn.cluster import KMeans

def kmeans(k, data):
    print("clustering kmeans...")
    return KMeans(
        n_clusters=k,
        max_iter=100,
        n_init=1,
    ).fit(data)
