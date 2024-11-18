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

def show_dendrogram(model, **kwargs):
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
    nodes = []
    allnodes = [{}]*(n_samples+len(model.children_))
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                counts_by_labels[i, labels[child_idx]] += 1
                # leaf node
                current_count += 1
                count_by_categ = [0] * n_clusters
                count_by_categ[labels[child_idx]] = 1
                allnodes[child_idx.item()] = {"node_id":child_idx.item(),"parent":i+n_samples, "count_by_true_categ": count_by_categ, "true_category": labels[child_idx].item()}
            else:
                current_count += counts[child_idx - n_samples]
                for label in range(0, n_clusters, 1):
                    counts_by_labels[i, label] += counts_by_labels[child_idx - n_samples, label]
        counts[i] = current_count
        node_dict = {"node_id": i+n_samples, "parent":None, "distance": model.distances_[i].item()}
        nodes.append(node_dict)
        allnodes[i+n_samples] = node_dict

    for i, node in enumerate(nodes):
        hasLeafChild = False
        children = []
        for child_idx in model.children_[i]:
            if child_idx >= n_samples:
                # set parent to this child
                allnodes[child_idx]["parent"] = node["node_id"]
                children.append(allnodes[child_idx])
            else:
                hasLeafChild = True
        if not hasLeafChild:
            node["children"] = children
        node["count_by_true_categ"] = counts_by_labels[node["node_id"]-n_samples].tolist()

    root_idx = [value for value in nodes if value["parent"] is None]
    return root_idx[0]