from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from config.constants import PredictionTarget
from models.base import BaseModel


class LSTMNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: List[int], dropout_rate: float, output_dim: int = 1):
        super(LSTMNetwork, self).__init__()

        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # First LSTM layer
        self.lstm_layers.append(nn.LSTM(input_dim, hidden_dim[0], batch_first=True,
                                        return_sequences=len(hidden_dim) > 1))
        self.dropout_layers.append(nn.Dropout(dropout_rate))

        # Additional LSTM layers
        for i in range(1, len(hidden_dim)):
            return_sequences = i < len(hidden_dim) - 1
            self.lstm_layers.append(nn.LSTM(hidden_dim[i - 1], hidden_dim[i], batch_first=True,
                                            return_sequences=return_sequences))
            self.dropout_layers.append(nn.Dropout(dropout_rate))

        # Output layer
        self.fc = nn.Linear(hidden_dim[-1], output_dim)
        self.sigmoid = nn.Sigmoid()  # For classification

    def forward(self, x, apply_sigmoid=False):
        # Forward pass through LSTM layers
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            x, _ = lstm(x)
            x = dropout(x)

        # Output layer
        out = self.fc(x)

        # Apply sigmoid if classification
        if apply_sigmoid:
            out = self.sigmoid(out)

        return out


class LSTMModel(BaseModel):
    """LSTM neural network model for time series forecasting using PyTorch."""

    def __init__(self, config: Dict):
        super().__init__(config)

        self.target_type = config.get('model', {}).get('prediction_target', 'direction')
        self.is_classifier = self.target_type in [PredictionTarget.DIRECTION.value,
                                                  PredictionTarget.VOLATILITY.value]

        # LSTM specific parameters
        self.sequence_length = config.get('model', {}).get('sequence_length', 10)
        self.batch_size = config.get('model', {}).get('batch_size', 32)
        self.epochs = config.get('model', {}).get('epochs', 50)

        # Default hyperparameters
        self.hyperparams = {
            'lstm_units': [128, 64],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': 10
        }

        # Update with user-provided hyperparameters if available
        if 'hyperparameters' in config.get('model', {}) and 'lstm' in config['model']['hyperparameters']:
            self.hyperparams.update(config['model']['hyperparameters']['lstm'])

        # Feature scaler for LSTM
        self.feature_scaler = StandardScaler()

        # Model will be built in fit() when input dimensions are known
        self.model = None
        self.history = {'train_loss': [], 'val_loss': []}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _build_model(self, input_dim: int) -> None:
        """Build the LSTM model architecture."""
        self.model = LSTMNetwork(
            input_dim=input_dim,
            hidden_dim=self.hyperparams['lstm_units'],
            dropout_rate=self.hyperparams['dropout_rate'],
            output_dim=1
        ).to(self.device)

        # Define optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hyperparams['learning_rate']
        )

        # Define loss function
        if self.is_classifier:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

    def _prepare_sequences(self, data: np.ndarray) -> np.ndarray:
        """Convert array to LSTM input sequences."""
        sequences = []

        # Create sequences
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])

        return np.array(sequences)

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None
    ) -> None:
        """Train the model on the provided data."""
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)

        # Prepare sequences
        X_train_seq = self._prepare_sequences(X_train_scaled)

        # Prepare target - offset to match sequence
        y_train_seq = y_train.iloc[self.sequence_length - 1:].values

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq.reshape(-1, 1)).to(self.device)

        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.hyperparams['batch_size'],
            shuffle=True
        )

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.feature_scaler.transform(X_val)
            X_val_seq = self._prepare_sequences(X_val_scaled)
            y_val_seq = y_val.iloc[self.sequence_length - 1:].values

            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq.reshape(-1, 1)).to(self.device)

            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.hyperparams['batch_size'],
                shuffle=False
            )

        # Build model if not already built
        if self.model is None:
            input_dim = X_train.shape[1]
            self._build_model(input_dim)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.hyperparams['epochs']):
            # Training
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X, apply_sigmoid=False)
                outputs = outputs.squeeze(-1)

                # Calculate loss
                loss = self.criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            epoch_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(epoch_train_loss)

            # Validation
            if val_loader:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X, apply_sigmoid=False)
                        outputs = outputs.squeeze(-1)
                        loss = self.criterion(outputs, batch_y)
                        val_losses.append(loss.item())

                epoch_val_loss = np.mean(val_losses)
                self.history['val_loss'].append(epoch_val_loss)

                # Early stopping
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.hyperparams['patience']:
                        print(f"Early stopping at epoch {epoch + 1}")
                        # Restore best model
                        self.model.load_state_dict(best_model_state)
                        break

                print(
                    f"Epoch {epoch + 1}/{self.hyperparams['epochs']}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{self.hyperparams['epochs']}, Train Loss: {epoch_train_loss:.4f}")

        self.is_fitted = True

        # Store feature importances (not directly available for LSTM)
        self.feature_importance = {feature: 0 for feature in X_train.columns}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        # Scale features
        X_scaled = self.feature_scaler.transform(X)

        # Prepare sequences
        X_seq = self._prepare_sequences(X_scaled)

        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor, apply_sigmoid=self.is_classifier)
            predictions = outputs.cpu().numpy().flatten()

        # For binary classification, convert probabilities to class labels
        if self.is_classifier:
            predictions = (predictions > 0.5).astype(int)

        # Pad predictions to match original data length
        pad_length = len(X) - len(predictions)
        padded_predictions = np.pad(predictions, (pad_length, 0), 'constant', constant_values=np.nan)

        return padded_predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prediction probabilities for classification models."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        if not self.is_classifier:
            raise ValueError("predict_proba() is only available for classification models.")

        # Scale features
        X_scaled = self.feature_scaler.transform(X)

        # Prepare sequences
        X_seq = self._prepare_sequences(X_scaled)

        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # Get probabilities
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor, apply_sigmoid=True)
            probas = outputs.cpu().numpy().flatten()

        # Convert to binary classification format [P(0), P(1)]
        binary_probas = np.column_stack([1 - probas, probas])

        # Pad probabilities to match original data length
        pad_length = len(X) - len(binary_probas)
        padded_probas = np.pad(binary_probas, ((pad_length, 0), (0, 0)), 'constant', constant_values=np.nan)

        return padded_probas

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importances."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        # LSTM doesn't provide feature importance directly
        return self.feature_importance

    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        # Save PyTorch model
        model_path = filepath.replace('.joblib', '_pytorch.pt')
        torch.save(self.model.state_dict(), model_path)

        # Save scaler and other attributes
        import joblib
        save_dict = {
            'feature_scaler': self.feature_scaler,
            'hyperparams': self.hyperparams,
            'is_classifier': self.is_classifier,
            'sequence_length': self.sequence_length,
            'target_type': self.target_type,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted,
            'model_path': model_path,
            'history': self.history,
            'input_dim': next(self.model.parameters()).shape[1]
        }
        joblib.dump(save_dict, filepath)

    def load(self, filepath: str) -> None:
        """Load the model from disk."""
        import joblib

        # Load attributes
        save_dict = joblib.load(filepath)

        self.feature_scaler = save_dict['feature_scaler']
        self.hyperparams = save_dict['hyperparams']
        self.is_classifier = save_dict['is_classifier']
        self.sequence_length = save_dict['sequence_length']
        self.target_type = save_dict['target_type']
        self.feature_importance = save_dict['feature_importance']
        self.is_fitted = save_dict['is_fitted']
        self.history = save_dict.get('history', {'train_loss': [], 'val_loss': []})

        # Rebuild model
        input_dim = save_dict.get('input_dim')
        if input_dim:
            self._build_model(input_dim)

            # Load PyTorch model state
            model_path = save_dict['model_path']
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

    def get_hyperparameters(self) -> Dict:
        """Get hyperparameters used for training."""
        return self.hyperparams