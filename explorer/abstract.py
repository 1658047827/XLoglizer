import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class StateAbstractor:
    def __init__(self, state_num):
        self.state_num = state_num


class GMMAbstractor(StateAbstractor):
    def __init__(self, state_num):
        super().__init__(state_num)


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
