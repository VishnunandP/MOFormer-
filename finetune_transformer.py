import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model.utils import split_data, train, validate
from model.transformer import Transformer
from model.dataset import CustomDataset


class FineTune:
    def __init__(self, config, log_dir):
        self.config = config
        self.log_dir = log_dir

        # Load data
        try:
            self.data = pd.read_excel(self.config['dataset']['dataPath'])
            print(f"Dataset loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {self.config['dataset']['dataPath']}.")
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")

        # Ensure consistent indexing
        self.data.reset_index(drop=True, inplace=True)

        # Split data into train, validation, and test sets
        self.train_data, self.valid_data, self.test_data = split_data(
            self.data,
            self.config['dataset']['testRatio'],
            self.config['dataset']['validRatio'],
            randomSeed=self.config['dataset']['randomSeed']
        )

        print(
            f"Train size: {len(self.train_data)}, "
            f"Validation size: {len(self.valid_data)}, "
            f"Test size: {len(self.test_data)}"
        )

        # Separate features and labels
        train_features, train_labels = self.split_features_labels(self.train_data)
        valid_features, valid_labels = self.split_features_labels(self.valid_data)
        test_features, test_labels = self.split_features_labels(self.test_data)

        # Create datasets and data loaders
        self.train_dataset = CustomDataset(train_features.values, train_labels.values)
        self.valid_dataset = CustomDataset(valid_features.values, valid_labels.values)
        self.test_dataset = CustomDataset(test_features.values, test_labels.values)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['training']['batchSize'], shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.config['training']['batchSize'], shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config['training']['batchSize'], shuffle=False)

        # Initialize model
        # Initialize model
        self.model = Transformer(
            ntoken=self.config['model']['ntoken'],  # Set the vocabulary size
            d_model=self.config['model']['d_model'],  # Embedding dimension
            nhead=self.config['model']['nhead'],  # Number of attention heads
            d_hid=self.config['model']['d_hid'],  # Hidden layer size in the feedforward network
            nlayers=self.config['model']['nlayers'],  # Number of Transformer layers
            dropout=self.config['model']['dropout']  # Dropout probability
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learningRate']
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def split_features_labels(self, data):
        """
        Splits the dataset into features and labels.

        :param data: Pandas DataFrame
        :return: features (Pandas DataFrame), labels (Pandas Series)
        """
        # Assuming the label column is the last column
        features = data.iloc[:, :-1]
        labels = data.iloc[:, -1]
        return features, labels

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.config['training']['epochs']):
            print(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            train_loss = train(self.model, self.train_loader, self.optimizer, self.criterion, self.device)
            val_loss = validate(self.model, self.valid_loader, self.criterion, self.device)

            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best_model.pth'))

        print("Training complete.")

    def test(self):
        print("Testing the model...")
        test_loss = validate(self.model, self.test_loader, self.criterion, self.device)
        print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    # Example configuration
    config = {
        'dataset': {
            'dataPath': 'PS_Usable_Hydrogen_Storgae_Capaciti_GCMC (1).xlsx',
            'validRatio': 0.1,
            'testRatio': 0.1,
            'randomSeed': 42
        },
        'training': {
            'batchSize': 32,
            'learningRate': 0.001,
            'epochs': 20
        },
        'model': {
            'inputDim': 128,
            'hiddenDim': 256,
            'numHeads': 4,
            'numLayers': 2,
            'outputDim': 10
        }
    }

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    fine_tune = FineTune(config, log_dir)
    fine_tune.train()
    fine_tune.test()
