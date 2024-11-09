from flask import Flask
from flask_cors import CORS
import dataset
import textprocessing
import dimred
import clustering
import projection

app = Flask(__name__)
CORS(app) # allows cross origin to be called from localhost:3000, not recommended in production

dataset, true_k = dataset.get20newsgroups()
x_tfidf, vectorizer = textprocessing.get_tfidf(dataset.data)
x_lsa, lsa = dimred.lsa(x_tfidf)

kmeans = clustering.kmeans(true_k, x_lsa)

embs_cosine = projection.tsne_cosine(x_tfidf)
# embs_euclidean = projection.tsne_euclidean(x_lsa)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/getProjection")
def get_projection():
    return embs_cosine.tolist()