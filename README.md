
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
    python3 model.py
    ```

📊 Model Performance and Output Explanation
After running the model, you will see an output similar to this:

markdown

Copy

Edit
              precision    recall  f1-score   support

          Normal       1.00      0.93      0.96       307
         Anomaly       0.51      1.00      0.68        23

        accuracy                           0.93       330
       macro avg       0.76      0.96      0.82       330
    weighted avg       0.97      0.93      0.94       330

What does this mean?
Precision: Of all samples the model predicted as a certain class, how many were correct?

Normal precision (1.00) means the model’s predictions labeled as “Normal” are almost always correct.

Anomaly precision (0.51) means only about 51% of predicted anomalies were truly anomalies, so there are some false positives.

Recall: Of all actual samples in a class, how many did the model detect correctly?

Normal recall (0.93) means the model correctly identified 93% of normal data points.

Anomaly recall (1.00) means the model detected all actual anomalies (no false negatives).

F1-score: The harmonic mean of precision and recall, balancing both.

For normal data, the score is high (0.96), showing strong performance.

For anomalies, the score is lower (0.68), showing room for improvement.

Support: Number of true instances for each class (307 normal, 23 anomaly).

Accuracy: Overall, the model correctly classified 93% of the samples.

Summary
The model is very good at finding all anomalies (100% recall), meaning it catches everything unusual.

However, it has some false positives (precision 0.51) — it sometimes wrongly flags normal points as anomalies.

The ensemble approach aims to reduce these false positives while keeping recall high.
