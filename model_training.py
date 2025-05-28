# model_training.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the model
joblib.dump(model, "model/iris_model.pkl")
print("Model trained and saved in model/iris_model.pkl")
