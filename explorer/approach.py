import os
import torch
import logging
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from abstract import PCAReducer, KMeansAbstractor, GMMAbstractor

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

        self.reducer = PCAReducer(reduced_dim)
        self.abstractor = KMeansAbstractor(state_num)
        # self.abstractor = GMMAbstractor(state_num)

    @torch.no_grad()
    def profile(self, train_loader, topk):
        self.logger.info("Collect hidden vectors from RNN and train dataset")
        collect = defaultdict(list)
        for batch in train_loader:
            x = (
                batch["feature"]
                .view(-1, self.window_size, self.input_size)
                .to(self.device)
            )
            out, pred = self.model.profile(x)
            collect["session_id"].extend(batch["session_id"])
            collect["vectors"].append(out)
            collect["predictions"].append(pred)
            collect["inputs"].append(x)
            collect["labels"].append(batch["label"])

        vectors = torch.cat(collect["vectors"]).cpu().numpy()
        predictions = torch.cat(collect["predictions"])
        preds = torch.argmax(predictions, dim=2).cpu().numpy()
        inputs = torch.cat(collect["inputs"]).squeeze().cpu().numpy().astype(int)
        labels = torch.cat(collect["labels"]).cpu().numpy()

        topk_values, topk_indices = torch.topk(predictions[:, -1, :].squeeze(), topk)
        df = pd.DataFrame({
            "session_id": collect["session_id"],
            "input": inputs.tolist(),
            "label": labels.tolist(),
            "topk_pred": topk_indices.cpu().numpy().tolist(),
            "topk_value": topk_values.cpu().numpy().tolist(),
        })
        print(df)

        np.save(f"{file_dir}/cache/vectors.npy", vectors)
        np.save(f"{file_dir}/cache/preds.npy", preds)
        np.save(f"{file_dir}/cache/inputs.npy", inputs)
        np.save(f"{file_dir}/cache/labels.npy", labels)

        return df

    def dimension_reduction(self, vectors):
        samples = vectors.reshape(-1, vectors.shape[-1])
        #####################################
        # should I just use unique samples? #
        #####################################
        unique_samples = np.unique(samples, axis=0)
        self.reducer.fit(unique_samples)
        reduced = self.reducer.transform(samples)
        reduced_vectors = reduced.reshape(-1, self.window_size, self.reduced_dim)
        return reduced_vectors

    def state_abstraction(self, vectors):
        samples = vectors.reshape(-1, vectors.shape[-1])
        self.abstractor.fit(samples)
        labels = self.abstractor.predict(samples)
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
        data = [defaultdict(int) for _ in range(self.state_num + 1)]
        for state, eid in zip(traces.flat, inputs.flat):
            state_input[state][eid] += 1
            data[state][int(eid)] += 1

        np.save(f"{file_dir}/cache/state_input.npy", state_input)
        np.savetxt(f"{file_dir}/cache/state_input.txt", state_input, fmt="%d")
        json.dump(data, open(f"{file_dir}/cache/state_input.json", "w"))

        return state_input

    def gather_state_label_statistics(self, traces, preds):
        state_label = np.zeros((self.state_num + 1, self.num_labels), dtype=int)
        data = [defaultdict(int) for _ in range(self.state_num + 1)]
        for state, label in zip(traces[:, -1].flat, preds[:, -1].flat):
            state_label[state][label] += 1
            data[state][int(label)] += 1

        np.save(f"{file_dir}/cache/state_label.npy", state_label)
        np.savetxt(f"{file_dir}/cache/state_label.txt", state_label, fmt="%d")
        json.dump(data, open(f"{file_dir}/cache/state_label.json", "w"))

        return state_label

    def fidelity(self, traces, preds, state_label):
        state_pred = np.argmax(state_label, axis=1)
        pred = state_pred[traces[:, -1].squeeze()]
        rnn_pred = preds[:, -1].squeeze()
        count = np.sum(pred == rnn_pred)
        fdlt = count / len(traces)
        self.logger.info(f"Fidelity of abstraction model: {fdlt}")
        return fdlt

    def get_graph(self, transitions, state_input, threshold):
        norm_matrix = transitions / np.sum(transitions, axis=1, keepdims=True)
        mask = norm_matrix < threshold
        norm_matrix[mask] = 0
        G = nx.from_numpy_array(norm_matrix, create_using=nx.DiGraph)

        state_freq = np.sum(state_input, axis=1)
        base_node_size = 10
        s0_freq = np.mean(state_freq)
        state_freq[0] = s0_freq
        sizes = np.round(base_node_size * state_freq / s0_freq, 1)
        node_size = [sizes[id] for id in G.nodes]

        # print(G.edges(data=True))
        edge_widths = [attr["weight"] for _, _, attr in G.edges(data=True)]
        # pos = nx.circular_layout(G)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, width=edge_widths, node_size=node_size)
        plt.show()

        for id in G.nodes:
            G.nodes[id]["size"] = sizes[id]
        d = nx.json_graph.node_link_data(G)
        json.dump(d, open(f"{file_dir}/cache/force.json", "w"))
