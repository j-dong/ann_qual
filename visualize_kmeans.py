import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

import matplotlib.animation as animation

vectors = np.fromfile('G:/vectors/siftsmall/siftsmall_base.fvecs', np.float32)
dim = vectors.view(np.int32)[0]
print(dim)
num_vectors = len(vectors) // (dim + 1)
print(num_vectors)
vectors = np.lib.stride_tricks.as_strided(
        vectors[1:],
        (num_vectors, dim),
        ((dim + 1) * 4, 4)
)

viz = PCA(2)
viz_vecs = viz.fit_transform(vectors)

fig, ax = plt.subplots()

frames = []

for i in range(1, 21):
    print('showing frame', i)
    assign = np.fromfile(f'assign_{i}', dtype=np.int32)
    clusters = np.fromfile(f'clusters_{i}', dtype=np.float32)
    clusters = clusters.reshape((-1, dim))
    print(clusters[np.any(np.isnan(clusters), axis=1), :])
    print(len(set(assign)))
    scatter = ax.scatter(viz_vecs[:, 0], viz_vecs[:, 1], c=assign, marker='+')
    viz_clusters = viz.transform(clusters)
    centroids = ax.scatter(viz_clusters[:, 0], viz_clusters[:, 1], c=np.arange(clusters.shape[0]), marker='*', s=2)
    frames.append((scatter, centroids))

ani = animation.ArtistAnimation(fig, frames)

plt.show()
