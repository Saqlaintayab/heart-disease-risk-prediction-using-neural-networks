# heart-disease-risk-prediction-using-neural-networks
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/Users/saqlaintayab/Desktop/heart.csv')

# Display first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Preprocessing
# Handle missing values (if any)
data = data.dropna()

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Building the neural network
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print('Confusion Matrix:')
print(conf_matrix)

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC Score: {roc_auc:.4f}')

# Plotting Training History
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()

# Function to make predictions for new inputs
def predict_heart_disease(input_data):
    input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    prediction = model.predict(input_data_scaled)
    return 'Heart Disease' if prediction > 0.5 else 'No Heart Disease'

# Example prediction
new_patient_data = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]  # Example input
prediction = predict_heart_disease(new_patient_data)
print(f'Prediction for new patient data: {prediction}')

# Pie chart for actual classes
actual_counts = y_test.value_counts()
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.pie(actual_counts, labels=['No Heart Disease', 'Heart Disease'], autopct='%1.1f%%', startangle=140)
plt.title('Actual Class Distribution')

# Pie chart for predicted classes
predicted_counts = pd.Series(y_pred_classes.flatten()).value_counts()
plt.subplot(1, 2, 2)
plt.pie(predicted_counts, labels=['No Heart Disease', 'Heart Disease'], autopct='%1.1f%%', startangle=140)
plt.title('Predicted Class Distribution')

plt.show()
