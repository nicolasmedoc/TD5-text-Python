from sklearn.manifold import TSNE

def tsne_cosine(data):
    print("TSNE cosine...")
    model = TSNE(random_state=1, metric="cosine", init="random")
    return model.fit_transform(data)

def tsne_euclidean(data):
    print("TSNE euclidean...")
    model = TSNE(random_state=1)
    return model.fit_transform(data)
