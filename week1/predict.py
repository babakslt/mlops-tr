# predict.py
import pickle
import sys

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict (example: first sample)
sample = [[5.1, 3.5, 1.4, 0.2]]  # Iris setosa
pred = model.predict(sample)
print(f"Prediction: {pred[0]}")
