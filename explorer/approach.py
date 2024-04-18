import os
import torch
import logging
import joblib
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
        reduced_dim,
        state_num,
    ):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.reduced_dim = reduced_dim
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

    def dimension_reduction(self, reducer, vectors):
        samples = vectors.reshape(-1, vectors.shape[-1])
        reducer.fit(samples)
        reduced = reducer.transform(samples)
        reduced_vectors = reduced.reshape(-1, self.window_size, self.reduced_dim)
        return reduced_vectors

    def state_abstraction(self, abstractor, vectors):
        samples = vectors.reshape(-1, vectors.shape[-1])
        abstractor.fit(samples)
        labels = abstractor.predict(samples)
        traces = labels.reshape(-1, self.window_size)
        np.save(f"{file_dir}/cache/traces.npy", traces)
        # np.savetxt(f"{file_dir}/cache/traces.txt", traces, fmt="%d")
        return traces

    def get_transitions(self, traces):
        transitions = np.zeros((self.state_num + 1, self.state_num + 1), dtype=int)
        for trace in traces:
            last_state = 0
            for state in trace:
                transitions[last_state][state] += 1
                last_state = state
        np.save(f"{file_dir}/cache/transitions.npy", transitions)
        # np.savetxt(f"{file_dir}/cache/transitions.txt", transitions, fmt="%d")
        return transitions

    def gather_state_input_statistics(self, traces, inputs):
        state_input = np.zeros((self.state_num + 1, self.num_labels), dtype=int)
        for state, eid in zip(traces.flat, inputs.flat):
            state_input[state][eid] += 1
        np.save(f"{file_dir}/cache/state_input.npy", state_input)
        # np.savetxt(f"{file_dir}/cache/state_input.txt", state_input, fmt="%d")
        return state_input
    
    def gather_state_label_statistics(self, traces, preds):
        state_label = np.zeros((self.state_num + 1, self.num_labels), dtype=int)
        for state, label in zip(traces.flat, preds.flat):
            state_label[state][label] += 1
        np.save(f"{file_dir}/cache/state_label.npy", state_label)
        # np.savetxt(f"{file_dir}/cache/state_label.txt", state_label, fmt="%d")
        return state_label

    def get_graph(self, transitions, state_label, threshold):
        norm_matrix = transitions / np.sum(transitions, axis=1, keepdims=True)
        mask = norm_matrix < threshold
        norm_matrix[mask] = 0
        G = nx.from_numpy_array(norm_matrix, create_using=nx.DiGraph)

        state_freq = np.sum(state_label, axis=1)
        base_node_size = 10
        s0_freq = np.mean(state_freq)
        state_freq[0] = s0_freq
        sizes = np.round(base_node_size * state_freq / s0_freq, 1)
        node_size = [sizes[id] for id in G.nodes]

        print(G.edges(data=True))
        edge_widths = [attr["weight"] for _, _, attr in G.edges(data=True)]
        # pos = nx.circular_layout(G)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, width=edge_widths, node_size=node_size)
        plt.show()

        for id in G.nodes:
            G.nodes[id]["size"] = sizes[id]
        d = nx.json_graph.node_link_data(G)
        json.dump(d, open(f"{file_dir}/cache/force.json", "w"))

        