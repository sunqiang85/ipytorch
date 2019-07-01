import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


if __name__ == "__main__":

    # load data
    y=np.load('logs/mnist/sgd/targets.npy')
    X= np.load('logs/mnist/sgd/fc2.npy')



    # fit
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)


    plt.figure(figsize=(6, 5))

    # figure config
    target_ids = [c for c in range(10)]
    target_names = [str(c) for c in range(10)]
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    k=20


    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(X_2d[y == i, 0][:k], X_2d[y == i, 1][:k], c=c, label=label)
    plt.legend()
    plt.savefig('figs/mnist/tsne.png')
    plt.show()