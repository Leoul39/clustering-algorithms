import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k=3, max_iters=100, tol=1e-4):
    np.random.seed(42)
        
    n_samples, n_features = X.shape
    
    
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else centroids[i]
            for i in range(k)
        ])
        
        # Check convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
            
        centroids = new_centroids
        
    return labels


def dbscan(X, eps=0.5, min_pts=5):
    
    n_samples = X.shape[0]
    labels = np.full(n_samples, -1)  # initialize all as noise
    visited = np.zeros(n_samples, dtype=bool)
    cluster_id = 0
    
    def region_query(point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= eps)[0]
    
    def expand_cluster(point_idx, neighbors):
        labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            
            if not visited[neighbor]:
                visited[neighbor] = True
                new_neighbors = region_query(neighbor)
                
                if len(new_neighbors) >= min_pts:
                    neighbors = np.concatenate((neighbors, new_neighbors))
            
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
                
            i += 1
    
    for i in range(n_samples):
        if visited[i]:
            continue
            
        visited[i] = True
        neighbors = region_query(i)
        
        if len(neighbors) < min_pts:
            labels[i] = -1  # noise
        else:
            expand_cluster(i, neighbors)
            cluster_id += 1
    
    return labels


def hdbscan(X, min_samples=15, min_cluster_size=30, cluster_selection_epsilon=None):

    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    k = int(min_samples)

    # STEP 1: Compute kNN (exact, vectorized, chunked for memory)

    knn_idx = np.empty((n, k), dtype=np.int32)
    knn_dist = np.empty((n, k), dtype=np.float64)

    chunk = 1000  # prevents 10k x 10k full matrix allocation
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = X[start:end, None, :] - X[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)

        # exclude self
        for i in range(start, end):
            dist2[i - start, i] = np.inf

        idx_part = np.argpartition(dist2, k, axis=1)[:, :k]
        dist_part = np.take_along_axis(dist2, idx_part, axis=1)

        order = np.argsort(dist_part, axis=1)
        idx_sorted = np.take_along_axis(idx_part, order, axis=1)
        dist_sorted = np.sqrt(np.take_along_axis(dist_part, order, axis=1))

        knn_idx[start:end] = idx_sorted
        knn_dist[start:end] = dist_sorted

    # STEP 2: Core distances
    core_dist = knn_dist[:, -1]

    # STEP 3: Mutual reachability graph (sparse from kNN)
    U = np.repeat(np.arange(n), k)
    V = knn_idx.reshape(-1)
    D = knn_dist.reshape(-1)

    W = np.maximum(np.maximum(core_dist[U], core_dist[V]), D)

    # Remove duplicate edges (undirected)
    a = np.minimum(U, V)
    b = np.maximum(U, V)
    mask = a != b
    a, b, W = a[mask], b[mask], W[mask]

    key = a.astype(np.int64) * n + b
    order = np.argsort(key)
    key, a, b, W = key[order], a[order], b[order], W[order]

    keep = np.concatenate(([True], key[1:] != key[:-1]))
    a, b, W = a[keep], b[keep], W[keep]

    # STEP 4: Minimum Spanning Tree (Kruskal)
    order = np.argsort(W)
    a, b, W = a[order], b[order], W[order]

    parent = np.arange(n)
    size = np.ones(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mst_edges = []
    for u, v, w in zip(a, b, W):
        ru, rv = find(u), find(v)
        if ru != rv:
            if size[ru] < size[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            size[ru] += size[rv]
            mst_edges.append((u, v, w))
            if len(mst_edges) == n - 1:
                break

    mst_edges = np.array(mst_edges)
    weights = mst_edges[:, 2]

    # STEP 5: Cluster selection (flat extraction)
    if cluster_selection_epsilon is None:
        # automatic robust cut (90th percentile)
        eps_cut = np.percentile(weights, 90)
    else:
        eps_cut = cluster_selection_epsilon

    parent = np.arange(n)
    size = np.ones(n)

    def find2(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v, w in mst_edges:
        if w > eps_cut:
            break
        ru, rv = find2(int(u)), find2(int(v))
        if ru != rv:
            if size[ru] < size[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            size[ru] += size[rv]

    roots = np.array([find2(i) for i in range(n)])
    unique, inv = np.unique(roots, return_inverse=True)
    comp_sizes = np.bincount(inv)

    labels = -np.ones(n, dtype=int)
    cluster_id = 0
    for i, s in enumerate(comp_sizes):
        if s >= min_cluster_size:
            labels[inv == i] = cluster_id
            cluster_id += 1

    return labels

def plot_clusters(X, labels, title="Clusters"):
    plt.figure(figsize=(14, 8))
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        pts = X[labels == lbl]
        if lbl == -1:
            plt.scatter(pts[:, 0], pts[:, 1], c='black', label="noise", s=10)
        else:
            plt.scatter(pts[:, 0], pts[:, 1], label=f"cluster {lbl+1}", s=20)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.show()

def pairwise_distances(X):
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def silhouette_score(X, labels):
    mask = labels != -1
    X = X[mask]
    labels = labels[mask]

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None

    D = pairwise_distances(X)
    sil_scores = []

    for i in range(len(X)):
        same_cluster = labels == labels[i]
        other_clusters = labels != labels[i]

        a = np.mean(D[i, same_cluster]) if np.sum(same_cluster) > 1 else 0

        b = np.min([
            np.mean(D[i, labels == lbl])
            for lbl in unique_labels if lbl != labels[i]
        ])

        sil_scores.append((b - a) / max(a, b))

    return np.mean(sil_scores)