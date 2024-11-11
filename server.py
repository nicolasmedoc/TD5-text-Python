import dataset
import textprocessing
import dimred
import projection
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

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/getProjection")
def get_projection():
    return {'projection': proj_euclidean.tolist(), 'categories': dataset.target.tolist()}