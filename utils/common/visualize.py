import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_TSNE(total_features, total_labels, save_path) :
    features_embedded = TSNE(n_components=2, init='random', verbose=1, random_state=42).fit_transform(total_features)

    cdict = {0: 'black', 1: 'red', 2: 'blue', 3: 'green', 4: 'cyan', 5: 'magenta', 6: 'yellow'}

    fig, ax = plt.subplots(dpi=600)
    for label in np.unique(total_labels) :
        ix = np.where(total_labels == label)
        ax.scatter(features_embedded[ix,0], features_embedded[ix,1], c=cdict[label], label=label, s=1**2)
    ax.legend()
    plt.title("Defocused data T-SNE projection")
    plt.savefig(save_path)