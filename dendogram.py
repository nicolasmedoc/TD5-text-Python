import numpy as np
from scipy.cluster.hierarchy import  dendrogram
from matplotlib import pyplot as plt

def plot_dendrogram(model, **kwargs):
    # from https: // scikit - learn.org / stable / auto_examples / cluster / plot_agglomerative_dendrogram.html
    # Create linkage matrix and then plot the dendrogram
    # nb nodes in dendogram == n_samples - 1
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, color_threshold=0.5*max(model.distances_), **kwargs)

def show_dendogram(model, **kwargs):
    plot_dendrogram(model, **kwargs)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

def get_tree_dict(model, true_k=None, ground_truth=None):
    n_clusters=model.n_clusters_
    if true_k is not None:
        n_clusters = true_k
    labels = model.labels_
    if ground_truth is not None:
        labels=ground_truth
    counts = np.zeros(model.children_.shape[0])
    counts_by_labels = np.zeros((model.children_.shape[0],n_clusters))
    n_samples = len(labels)
    nodes=[]
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                counts_by_labels[i, labels[child_idx]] += 1
                # leaf node
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
                for label in range(0, n_clusters, 1):
                    counts_by_labels[i, label] += counts_by_labels[child_idx - n_samples, label]
        counts[i] = current_count
        nodes.append({"node_id": i+n_samples, "parent":None, "distance": model.distances_[i], "children": [{"node_id": child, "parent":i+n_samples, "children":[], "category": labels[child] if child < n_samples else None} for child in merge]})


    for i,node in enumerate(nodes):
        children = []
        for child in node["children"]:
            if child["node_id"] >= n_samples:
                # set parent to this child
                nodes[child["node_id"] - n_samples]["parent"] = node["node_id"]
                children.append(nodes[child["node_id"] - n_samples])
        node["children"] = children
        node["count_by_categ"] = counts_by_labels[i].tolist()

    root_idx = [value for value in nodes if value["parent"] is None]
    return root_idx[0]