import numpy as np

num_vectors = 1000000
dim = 128

num_clusters = 128

clusters = np.random.normal(size=(num_clusters, dim))
norm = np.linalg.norm(clusters, axis=1)
clusters *= 100.0 * norm

assignments = np.random.randint(0, num_clusters, size=(num_vectors,))

vectors = np.random.normal(size=(num_vectors, dim)) + clusters[assignments, :]

file_out = np.insert(vectors, 0, 0.0, axis=1).astype(np.float32)
file_out.view(np.int32)[:, 0] = dim

file_out.tofile('kmeans_test.fvecs')
