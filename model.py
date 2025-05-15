import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Sample dataset loader (replace with your real dataset)
def load_data():
    # Simulate with synthetic data: 0=normal, 1=anomaly
    X_normal = np.random.normal(0, 1, (1000, 20))
    X_anomaly = np.random.normal(4, 1, (100, 20))
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0]*1000 + [1]*100)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.3, random_state=42), scaler


# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Train one autoencoder on only normal data
def train_autoencoder(X_train, input_dim, hidden_dim=10, epochs=50):
    model = Autoencoder(input_dim, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X_train_tensor = torch.FloatTensor(X_train)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, X_train_tensor)
        loss.backward()
        optimizer.step()
    
    return model


# Calculate reconstruction error
def compute_errors(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        output = model(X_tensor)
        mse = ((X_tensor - output) ** 2).mean(dim=1).numpy()
    return mse


# Ensemble logic (majority vote)
def ensemble_predict(models, X, threshold):
    votes = []
    for model in models:
        errors = compute_errors(model, X)
        vote = (errors > threshold).astype(int)
        votes.append(vote)
    
    votes = np.array(votes)
    predictions = (votes.sum(axis=0) >= (len(models) // 2 + 1)).astype(int)
    return predictions


# Main
(X_train, X_test, y_train, y_test), scaler = load_data()

# Train on only normal data
X_train_normal = X_train[y_train == 0]

# Create ensemble of 3 autoencoders
ensemble_models = [
    train_autoencoder(X_train_normal, input_dim=X_train.shape[1], hidden_dim=8),
    train_autoencoder(X_train_normal, input_dim=X_train.shape[1], hidden_dim=10),
    train_autoencoder(X_train_normal, input_dim=X_train.shape[1], hidden_dim=12)
]

# Use one model to decide threshold
errors = compute_errors(ensemble_models[0], X_train_normal)
threshold = np.percentile(errors, 95)  # Adjust for sensitivity

# Predict
y_pred = ensemble_predict(ensemble_models, X_test, threshold)

# Evaluation
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

