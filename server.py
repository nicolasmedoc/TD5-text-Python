import dataset
import textprocessing
import dimred
import projection
import clustering
import dendogram
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
# allows cross origin to be called from localhost:3000
# not recommended in production
CORS(app)

# insert code for server initialization if needed
dataset, true_k = dataset.get20newsgroups()
x_tfidf, vectorizer = textprocessing.get_tfidf(dataset.data)
x_lsa, lsa = dimred.lsa(x_tfidf)
proj_euclidean = projection.tsne_euclidean(x_lsa)
# cluster_tree = clustering.agglomerative(true_k,x_lsa,"ward","euclidean")
cluster_tree = clustering.agglomerative(None,x_lsa,"ward","euclidean",distance_threshold=0)
# dendogram.show_dendogram(cluster_tree, truncate_mode="level", p=4)
dendogram.show_dendogram(cluster_tree, truncate_mode="none", p=3)
print("n_clusters = "+str(cluster_tree.n_clusters_) + "true_k="+ str(true_k))
# cluster_tree_dict = dendogram.get_tree_dict(cluster_tree)
cluster_tree_dict = dendogram.get_tree_dict(cluster_tree, true_k, dataset.target)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/getProjection")
def get_projection():
    return {'projection': proj_euclidean.tolist(), 'categories': dataset.target.tolist()}


@app.route("/getClusterTree")
def get_cluster_tree():
    return {"cluster_tree": cluster_tree_dict, "categories": dataset.target.tolist(), "distances": cluster_tree.distances_.tolist()}
    # return {"cluster_tree": cluster_tree_dict, "categories": cluster_tree.labels_.tolist()}
    # return {"cluster_tree": cluster_tree.children_.tolist(), "categories": cluster_tree.labels_.tolist()}