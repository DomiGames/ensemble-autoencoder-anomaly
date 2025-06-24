import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the dataset
df = pd.read_csv("creditcard.csv")

# Separate features and labels
X = df.drop(columns=["Class"])
y = df["Class"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train only on normal (non-fraudulent) data
X_train = X_train_full[y_train_full == 0]
y_train = y_train_full[y_train_full == 0]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 14),
            nn.ReLU(),
            nn.Linear(14, 7),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(7, 14),
            nn.ReLU(),
            nn.Linear(14, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

# Train a single autoencoder
def train_autoencoder(model, data, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        output = model(data)
        loss = criterion(output, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

# Ensemble training
def train_ensemble(num_models, input_dim, data):
    ensemble = []
    for i in range(num_models):
        model = Autoencoder(input_dim)
        trained = train_autoencoder(model, data)
        ensemble.append(trained)
    return ensemble

# Compute reconstruction errors
def compute_errors(model, data):
    model.eval()
    with torch.no_grad():
        reconstructed = model(data)
        errors = torch.mean((reconstructed - data) ** 2, dim=1)
    return errors.cpu().numpy()

# Ensemble prediction based on average error
def ensemble_predict(models, data, threshold):
    all_errors = np.array([compute_errors(m, data) for m in models])
    avg_error = np.mean(all_errors, axis=0)
    return (avg_error > threshold).astype(int)

# Train the ensemble
input_dim = X_train.shape[1]
ensemble_models = train_ensemble(num_models=3, input_dim=input_dim, data=X_train_tensor)

# Get average reconstruction errors for training data (normal only)
train_errors = np.mean([compute_errors(m, X_train_tensor) for m in ensemble_models], axis=0)

# Set threshold (adjust to trade off precision/recall)
threshold = np.percentile(train_errors, 97)  # Make stricter for better precision

# Predict on test set
y_pred = ensemble_predict(ensemble_models, X_test_tensor, threshold)

# Report
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))


