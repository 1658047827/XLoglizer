import torch
import logging


class Trainer:
    def __init__(
        self, 
        model, 
        device, 
        optimizer, 
        criterion, 
        window_size, 
        input_size,
        save_path,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.window_size = window_size
        self.input_size = input_size
        self.save_path = save_path
        self.logger = logging.getLogger("loglizer")

    def fit(self, train_loader, valid_loader, epoches=10):
        best_loss = float("inf")
        for epoch in range(1, epoches + 1):
            self.train(epoch, train_loader)
            loss = self.validate(valid_loader)
            if loss < best_loss:
                best_loss = loss
                self.save_model()

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
        self.logger.info(f"Train epoch {epoch}, loss {epoch_loss}")

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
        self.logger.info(f"Validate loss {valid_loss}")
        return valid_loss

    def save_model(self):
        self.logger.info(f"Save model to {self.save_path}")
        torch.save(self.model.state_dict(), self.save_path)

    def load_model(self, load_path):
        self.logger.info(f"Load model from {load_path}")
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))