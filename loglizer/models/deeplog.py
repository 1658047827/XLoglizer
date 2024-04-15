import torch
import torch.nn as nn
import torch.nn.functional as F


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

    @torch.no_grad()
    def profile(self, input):
        """
        Args:
            input: tensor of shape (batch_size, window_size, input_size)

        Returns:
            out: tensor of shape (batch_size, window_size, hidden_size)
            pred: tensor of shape (batch_size, window_size, num_labels)
        """
        out, _ = self.lstm(input)
        repr = self.fc(out)
        pred = F.softmax(repr, dim=-1)
        return out, pred