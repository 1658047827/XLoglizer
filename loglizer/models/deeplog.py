import torch.nn as nn


class DeepLog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_labels):
        super(DeepLog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, input):
        out, _ = self.lstm(input)
        out = self.fc(out[:, -1, :])
        return out

    def profile(self, input):
        pass