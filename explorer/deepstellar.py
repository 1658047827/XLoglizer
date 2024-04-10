import os
import torch
import logging
import joblib
import numpy as np
from sklearn.decomposition import PCA

file_dir = os.path.dirname(os.path.abspath(__file__))


class DeepStellar:
    def __init__(
        self,
        model,
        device,
        train_loader,
        window_size,
        input_size,
        pca_components,
        state_num,
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.window_size = window_size
        self.input_size = input_size
        self.pca_components = pca_components
        self.state_num = state_num
        self.logger = logging.getLogger("explorer")
        self.pca = None
        self.vectors = None

    @torch.no_grad()
    def profile(self):
        self.logger.info("Collect hidden vectors from RNN and train dataset")

        tensor_list = []

        for batch in self.train_loader:
            x = batch["feature"].view(-1, self.window_size, self.input_size).to(self.device)
            out, pred = self.model.profile(x)
            tensor_list.append(out)

        vecs = torch.cat(tensor_list, dim=0)
        hidden_size = vecs.size(-1)
        self.vectors = vecs.view(-1, hidden_size).cpu().numpy()

    def pca_fit(self, load_cache=False):
        if load_cache:
            self.logger.info(f"Load cached PCA model from {file_dir}/cache/pca_model.joblib")
            self.pca = joblib.load(f"{file_dir}/cache/pca_model.joblib")
        else:
            self.logger.info("Fit PCA model with hidden vectors")
            self.pca = PCA(self.pca_components)
            self.pca.fit(self.vectors)
            self.logger.info(f"Save PCA model to {file_dir}/cache/pca_model.joblib")
            joblib.dump(self.pca, f"{file_dir}/cache/pca_model.joblib")

    def pca_transform(self, load_cache=False):
        if load_cache:
            self.logger.info(f"Load cached reduced vectors from {file_dir}/cache/reduced_vectors.npy")
            self.reduced_vecs = np.load(f"{file_dir}/cache/reduced_vectors.npy")
        else:
            self.logger.info("Apply dimensionality reduction to hidden vectors")
            self.reduced_vecs = self.pca.transform(self.vectors)
            self.logger.info(f"Save reduced vectors to {file_dir}/cache/reduced_vectors.npy")
            np.save(f"{file_dir}/cache/reduced_vectors.npy", self.reduced_vecs)
        
