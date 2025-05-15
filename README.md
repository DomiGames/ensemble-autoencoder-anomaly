
# Ensemble Autoencoder for Anomaly Detection

This project implements an **ensemble of autoencoders** to detect anomalies and reduce false positives in systems like fraud detection, intrusion detection, or industrial monitoring.

## 📊 What It Does
- Learns to reconstruct normal data
- Detects anomalies based on reconstruction errors
- Uses ensemble methods to improve robustness and reduce false alarms

## 🧠 Key Features
- Multiple autoencoders for robustness
- Anomaly threshold based on reconstruction error percentiles
- Classification report to evaluate performance

## 📁 Project Structure
ensemble-autoencoder-anomaly/

├── model.py # Main training and testing code

├── requirements.txt # Python dependencies

└── README.md # Project description

## 🚀 How to Run

1. Clone the repo:
    ```bash
    git clone https://github.com/DomiGames/ensemble-autoencoder-anomaly.git
    cd ensemble-autoencoder-anomaly
    ```

2. Create a virtual environment (optional):
   ```bash
    python3 -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```

5. Run the model:
    ```bash
    python model.py
    ```


EXAMPLE OUTPUT
              precision    recall  f1-score   support

      Normal       1.00      0.93      0.96       307
     Anomaly       0.51      1.00      0.68        23

    accuracy                           0.93       330
   macro avg       0.76      0.96      0.82       330
weighted avg       0.97      0.93      0.94       330
    
