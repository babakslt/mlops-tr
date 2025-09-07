# model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Train simple model
iris = load_iris()
clf = RandomForestClassifier()
clf.fit(iris.data, iris.target)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("✅ Model trained and saved!")
