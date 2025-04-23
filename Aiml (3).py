import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load sample dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Split dataset into features and target
X = df.drop('target', axis=1)
y = df['target']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train diagnostic model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Analyze loss function
y_pred_proba = model.predict_proba(X_test)
loss = log_loss(y_test, y_pred_proba)
print("Log Loss:", loss)

# Plot loss function
loss_values = []
for i in range(1, len(X_test)+1):
    loss_values.append(log_loss(y_test[:i], model.predict_proba(X_test[:i])))
plt.plot(loss_values)
plt.xlabel('Number of Samples')
plt.ylabel('Log Loss')
plt.title('Log Loss Over Samples')
plt.show()
