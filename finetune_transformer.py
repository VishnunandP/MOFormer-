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
        train_idx, valid_idx, test_idx = split_data(
            self.data,
            valid_ratio=self.config['dataset']['validRatio'],
            test_ratio=self.config['dataset']['testRatio'],
            random_seed=self.config['dataset']['randomSeed']
        )

        self.train_data = self.data.iloc[train_idx]
        self.valid_data = self.data.iloc[valid_idx]
        self.test_data = self.data.iloc[test_idx]

        print(
            f"Train size: {len(self.train_data)}, "
            f"Validation size: {len(self.valid_data)}, "
            f"Test size: {len(self.test_data)}"
        )

        # Create datasets and data loaders
        self.train_dataset = CustomDataset(self.train_data)
        self.valid_dataset = CustomDataset(self.valid_data)
        self.test_dataset = CustomDataset(self.test_data)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['training']['batchSize'], shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.config['training']['batchSize'], shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config['training']['batchSize'], shuffle=False)

        # Initialize model
        self.model = Transformer(
            input_dim=self.config['model']['inputDim'],
            hidden_dim=self.config['model']['hiddenDim'],
            num_heads=self.config['model']['numHeads'],
            num_layers=self.config['model']['numLayers'],
            output_dim=self.config['model']['outputDim']
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learningRate']
        )
        self.criterion = torch.nn.CrossEntropyLoss()

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
