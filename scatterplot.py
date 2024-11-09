import matplotlib.pyplot as plt

def show(x,labels):
    print("building scatterplot...")
    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots()

    ax.scatter(x[:,0], x[:,1], c=labels, cmap='Paired')
    plt.show()
