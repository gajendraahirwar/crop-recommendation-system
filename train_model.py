# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Step 2: Dataset load 
data = pd.read_csv("crop_data.csv")

print("Dataset ke pehle 5 rows:")
print(data.head())

print("\nDataset mein columns:", data.columns.tolist())
print("Total rows:", len(data))

# Step 3: Separate Input (X) and Output (y)
# X = soil/weather values, y = crop name
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Step 4: Split data into Train and Test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 6: check model accuracy on test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Step 7: save the trained model
with open("crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel successfully saved (crop_model.pkl)")