import torch
import logging


class Trainer:
    def __init__(self, model, device, optimizer, criterion):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

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
        for batch_input in train_loader:
            x = batch_input["x"].to(self.device)
            y = batch_input["y"].to(self.device)
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        logging.info(f"Train epoch {epoch}, loss {epoch_loss}")

    def validate(self, valid_loader):
        self.model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_input in valid_loader:
                x = batch_input["x"].to(self.device)
                y = batch_input["y"].to(self.device)
                output = self.model(x)
                valid_loss += self.criterion(output, y).item()
        valid_loss /= len(valid_loader)
        logging.info(f"Validate loss {valid_loss}")
        return valid_loss
