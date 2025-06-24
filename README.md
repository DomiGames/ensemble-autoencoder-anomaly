
# 🧠 Ensemble Autoencoder for Anomaly Detection

This project implements an **ensemble of autoencoders** to detect anomalies and reduce false positives in systems such as **fraud detection**, **intrusion detection**, or **industrial monitoring**. The system was trained and evaluated using the **Kaggle Credit Card Fraud Detection dataset**, a real-world dataset with highly imbalanced classes.

---

## 📊 What It Does
- Learns to reconstruct **normal** data using autoencoders
- Detects **anomalies** based on high reconstruction error
- Uses an **ensemble of models** to improve robustness and reduce false alarms
- Evaluates performance using precision, recall, and F1-score

---

## 🧠 Key Features
- Multiple autoencoders trained independently for ensemble voting
- Threshold-based anomaly detection using reconstruction error percentiles
- Real-world dataset (creditcard.csv from Kaggle)
- Evaluation report using `sklearn.metrics.classification_report`

---

## 📁 Project Structure

```
ensemble-autoencoder-anomaly/
├── creditcard.csv       # Real dataset from Kaggle (must be downloaded manually)
├── model.py             # Main training and testing code
├── requirements.txt     # Python dependencies
└── README.md            # Project description and instructions
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/DomiGames/ensemble-autoencoder-anomaly.git
cd ensemble-autoencoder-anomaly
```

### 2. (Optional) Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset  
Download [creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and place it in the project folder.

### 5. Run the model
```bash
python3 model.py
```

---

## 📊 Model Performance and Output Explanation

After running the model, you will see an output like this:

```
              precision    recall  f1-score   support

      Normal       1.00      0.93      0.96       307
     Anomaly       0.51      1.00      0.68        23

    accuracy                           0.93       330
   macro avg       0.76      0.96      0.82       330
weighted avg       0.97      0.93      0.94       330
```

### 🔍 What This Means:

- **Precision**: How many predicted anomalies were actually anomalies?
  - *Normal precision (1.00)*: All predictions labeled “Normal” were correct.
  - *Anomaly precision (0.51)*: Only 51% of predicted anomalies were real anomalies → some false alarms.

- **Recall**: How many true anomalies were detected?
  - *Normal recall (0.93)*: 93% of actual normal data was correctly classified.
  - *Anomaly recall (1.00)*: 100% of actual anomalies were caught (no false negatives).

- **F1-score**: Balance between precision and recall.
  - High for normal, moderate for anomaly.

- **Support**: Number of real samples in each class.

- **Accuracy (93%)**: Overall, the model performed very well.

---

### ✅ Summary

- The model **catches all anomalies** (high recall).
- It has **some false positives**, but this can be reduced by tweaking the threshold or using stricter voting rules.
- The **ensemble method improves generalization** and helps reduce the chance of overfitting to noise.
