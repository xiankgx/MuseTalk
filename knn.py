import numpy as np

import faiss


class KNearestNeighbor:

    def __init__(self, d: int):
        self.d = d
        self._fitted = False

    def fit(self, X: np.ndarray):
        self.index = faiss.IndexFlatL2(self.d)
        self.index.add(X)
        self._fitted = True 
        return self

    def predict(self, q: np.ndarray, k: int=1, return_distances: bool=False):
        if not self._fitted:
            raise Exception("Not fitted.")
        # distances, nearest neighbor indices
        D, I = self.index.search(q, k)
        if return_distances:
            return I.tolist(), D.tolist()
        return I.tolist()


if __name__ == "__main__":
    import numpy as np

    K = np.random.rand(100, 10 * 384)
    K = K.reshape(K.shape[0], -1)
    d = K.shape[-1]

    retriever = KNearestNeighbor(d)
    retriever = retriever.fit(K)
    I = retriever.predict(K[:1], k=2)
    print(f"I: {I}")