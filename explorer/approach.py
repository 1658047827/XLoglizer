import os
import torch
import logging
import joblib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from collections import defaultdict

file_dir = os.path.dirname(os.path.abspath(__file__))


class DeepStellar:
    def __init__(
        self,
        model,
        device,
        window_size,
        input_size,
        hidden_size,
        num_labels,
        pca_dim,
        state_num,
    ):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.pca_dim = pca_dim
        self.state_num = state_num
        self.logger = logging.getLogger("explorer")

    @torch.no_grad()
    def profile(self, train_loader):
        self.logger.info("Collect hidden vectors from RNN and train dataset")
        collect = defaultdict(list)
        for batch in train_loader:
            x = batch["feature"].view(-1, self.window_size, self.input_size).to(self.device)
            out, pred = self.model.profile(x)
            collect["vectors"].append(out)
            collect["predictions"].append(pred)
            collect["inputs"].append(x)
            collect["labels"].append(batch["label"])

        vectors = torch.cat(collect["vectors"]).cpu().numpy()
        predictions = torch.cat(collect["predictions"])
        preds = torch.argmax(predictions, dim=2).cpu().numpy()
        inputs = torch.cat(collect["inputs"]).squeeze().cpu().numpy().astype(int)
        labels = torch.cat(collect["labels"]).cpu().numpy()

        np.save(f"{file_dir}/cache/vectors.npy", vectors)
        np.save(f"{file_dir}/cache/preds.npy", preds)
        np.save(f"{file_dir}/cache/inputs.npy", inputs)
        np.save(f"{file_dir}/cache/labels.npy", labels)

    def pca_fit(self, load_cache=False):
        if load_cache:
            self.logger.info(f"Load cached PCA model from {file_dir}/cache/pca.joblib")
            self.pca = joblib.load(f"{file_dir}/cache/pca.joblib")
        else:
            self.logger.info("Fit PCA model with hidden vectors")
            
            self.pca = PCA(self.pca_dim)
            vecs = self.vectors.reshape(-1, self.hidden_size)
            self.pca.fit(vecs)

            self.logger.info(f"Save PCA model to {file_dir}/cache/pca.joblib")
            joblib.dump(self.pca, f"{file_dir}/cache/pca.joblib")

    def pca_transform(self, load_cache=False):
        if load_cache:
            self.logger.info(f"Load cached reduced vectors from {file_dir}/cache/reduced_windows.npy")
            self.reduced_windows = np.load(f"{file_dir}/cache/reduced_windows.npy")
        else:
            self.logger.info("Apply dimensionality reduction to hidden vectors")

            vecs = self.vectors.reshape(-1, self.hidden_size)
            reduced_vecs = self.pca.transform(vecs)
            self.reduced_windows = reduced_vecs.reshape(-1, self.window_size, self.pca_dim)

            self.logger.info(f"Save reduced windows to {file_dir}/cache/reduced_windows.npy")
            np.save(f"{file_dir}/cache/reduced_windows.npy", self.reduced_windows)

    def gmm_fit(self, load_cache=False):
        if load_cache:
            self.logger.info(f"Load cached GMM model from {file_dir}/cache/gmm.joblib")
            self.clustering = joblib.load(f"{file_dir}/cache/gmm.joblib")
        else:
            self.logger.info("Fit GMM model with reduced hidden vectors")

            self.clustering = GaussianMixture(self.state_num, covariance_type="diag")
            vecs = self.reduced_windows.reshape(-1, self.pca_dim)
            self.clustering.fit(vecs)

            bic = self.clustering.bic(vecs)

            self.logger.info(f"Save GMM model to {file_dir}/cache/gmm.joblib")
            joblib.dump(self.clustering, f"{file_dir}/cache/gmm.joblib")

    def gmm_predict(self, load_cache=False):
        vecs = self.reduced_windows.reshape(-1, self.pca_dim)
        self.traces = self.clustering.predict(vecs).reshape(-1, self.window_size) + 1

    def state_abstraction(self, abstractor, vectors):
        samples = vectors.reshape(-1, vectors.shape[-1])
        abstractor.fit(samples)
        labels = abstractor.predict(samples)
        traces = labels.reshape(-1, self.window_size)
        np.save(f"{file_dir}/cache/traces.npy", traces)
        np.savetxt(f"{file_dir}/cache/traces.txt", traces, fmt="%d")
        return traces

    def get_transitions(self, traces):
        transitions = np.zeros((self.state_num + 1, self.state_num + 1), dtype=int)
        for trace in traces:
            last_state = 0
            for state in trace:
                transitions[last_state][state] += 1
                last_state = state
        np.save(f"{file_dir}/cache/transitions.npy", transitions)
        np.savetxt(f"{file_dir}/cache/transitions.txt", transitions, fmt="%d")
        return transitions

    def gather_state_input_statistics(self, traces, inputs):
        state_input = np.zeros((self.state_num + 1, self.num_labels), dtype=int)
        for state, eid in zip(traces.flat, inputs.flat):
            state_input[state][eid] += 1
        np.save(f"{file_dir}/cache/state_input.npy", state_input)
        np.savetxt(f"{file_dir}/cache/state_input.txt", state_input, fmt="%d")
        return state_input
    
    def gather_state_label_statistics(self, traces, preds):
        state_label = np.zeros((self.state_num + 1, self.num_labels), dtype=int)
        for state, label in zip(traces.flat, preds.flat):
            state_label[state][label] += 1
        np.save(f"{file_dir}/cache/state_label.npy", state_label)
        np.savetxt(f"{file_dir}/cache/state_label.txt", state_label, fmt="%d")
        return state_label

    def draw(self, transitions, state_label):
        norm_matrix = transitions / np.linalg.norm(transitions, axis=1, keepdims=True)
        G = nx.from_numpy_array(norm_matrix, create_using=nx.DiGraph, edge_attr="width")

        state_freq = np.sum(state_label, axis=1)
        base_node_size = 500
        s0_freq = np.mean(state_freq)
        state_freq[0] = s0_freq
        node_size = np.round(base_node_size * state_freq / s0_freq, 1)

        print(node_size)

        # print(G.edges(data=True))
        edge_widths = [attr["width"] for _, _, attr in G.edges(data=True)]
        # pos = nx.circular_layout(G)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, width=edge_widths, node_size=node_size)
        plt.show()

        