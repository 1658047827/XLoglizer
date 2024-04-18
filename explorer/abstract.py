import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class DimensionReducer:
    def __init__(self, reduced_dim) -> None:
        self.reduced_dim = reduced_dim


class PCAReducer(DimensionReducer):
    def __init__(self, reduced_dim) -> None:
        super().__init__(reduced_dim)
        self.pca = PCA(reduced_dim)
        self.logger = logging.getLogger("explorer")

    def fit(self, samples):
        self.logger.info("Fit PCA reduction")
        self.pca.fit(samples)

    def transform(self, samples):
        self.logger.info("Apply dimension reduction to hidden vectors")
        reduced_vecs = self.pca.transform(samples)
        return reduced_vecs


class StateAbstractor:
    def __init__(self, state_num):
        self.state_num = state_num


class GMMAbstractor(StateAbstractor):
    def __init__(self, state_num):
        super().__init__(state_num)
        self.gmm = GaussianMixture(state_num)
        self.logger = logging.getLogger("explorer")

    def fit(self, samples):
        self.logger.info("Fit GMM clustering")
        num_extract = len(samples) // 10
        random_indices = np.random.choice(len(samples), num_extract, replace=False)
        random_samples = samples[random_indices]
        self.gmm.fit(random_samples)
        bic = self.gmm.bic(random_samples)
        self.logger.info(f"GMM Bayesian information criterion: {bic}")

    def predict(self, samples):
        self.logger.info("Predict cluster index for each sample")
        labels = self.gmm.predict(samples) + 1
        return labels


class KMeansAbstractor(StateAbstractor):
    def __init__(self, state_num):
        super().__init__(state_num)
        self.kmeans = KMeans(n_clusters=state_num)
        self.logger = logging.getLogger("explorer")

    def fit(self, samples):
        self.logger.info("Fit KMeans clustering")
        self.kmeans.fit(samples)

    def predict(self, samples):
        self.logger.info("Predict cluster index for each sample")
        labels = self.kmeans.predict(samples) + 1
        return labels
