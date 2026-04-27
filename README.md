# Deep Learning Project Dashboard (Streamlit)

A Streamlit web dashboard that visualizes multiple deep learning experiments (Regression, Classification, Forecasting, and Anomaly Detection).  
The app loads pre-saved **model parameters**, **model architecture (JSON)**, and **performance plots** from local folders and renders them in a clean UI.

---

## Features

- **4 Tabs**
  - 📈 Regression
  - 🧠 Classification
  - 📉 Forecasting
  - 🕵️ Anomaly Detection (Autoencoder)

- **Displays**
  - Dataset overview (samples, features, etc.)
  - Training configuration (epochs, batch size, optimizer, loss, learning rate)
  - Model architecture visualization using **Graphviz**
  - Performance figures (PNG)
  - Anomaly detection results (AUC, Precision, Recall, F1, Threshold, Quantile)

