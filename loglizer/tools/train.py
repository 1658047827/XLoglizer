import torch
import logging


class Trainer:
    def __init__(self, model, device, optimizer, criterion, window_size, input_size):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.window_size = window_size
        self.input_size = input_size

    def fit(self, train_loader, valid_loader, epoches=10):
        best_loss = float("inf")
        for epoch in range(1, epoches + 1):
            self.train(epoch, train_loader)
            loss = self.validate(valid_loader)
            if loss < best_loss:
                best_loss = loss

    def train(self, epoch, train_loader):
        self.model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            x = batch["feature"].view(-1, self.window_size, self.input_size).to(self.device)
            y = batch["label"].to(self.device)
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        logging.getLogger("loglizer").info(f"Train epoch {epoch}, loss {epoch_loss}")

    def validate(self, valid_loader):
        self.model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                x = batch["feature"].view(-1, self.window_size, self.input_size).to(self.device)
                y = batch["label"].to(self.device)
                output = self.model(x)
                valid_loss += self.criterion(output, y).item()
        valid_loss /= len(valid_loader)
        logging.getLogger("loglizer").info(f"Validate loss {valid_loss}")
        return valid_loss
