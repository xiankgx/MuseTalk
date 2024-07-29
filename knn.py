import numpy as np

import faiss


class KNearestNeighbor:

    def __init__(self, d: int):
        index = faiss.IndexFlatL2(d)
        self.index = index

    def fit(self, X: np.ndarray):
        self.index.add(X)
        return self

    def predict(self, q: np.ndarray, k:int=1):
        D, I = self.index.search(q, k)
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