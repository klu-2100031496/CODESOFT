# Step 1: Load the Dataset
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable

# Step 2: Explore the Dataset (optional)
print(iris.DESCR)  # Description of the dataset
print(iris.feature_names)  # Names of the features
print(iris.target_names)  # Names of the target classes

# Step 3: Preprocess the Data (No preprocessing required for this dataset)

# Step 4: Split the Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Select a Model
from sklearn.linear_model import LogisticRegression  # Example model, you can try others as well

model = LogisticRegression(max_iter=1000)  # Initialize the Logistic Regression model

# Step 6: Train the Model
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 8: Tune Hyperparameters (Optional, depending on the model chosen)

# Step 9: Make Predictions (Optional)
# Example prediction
new_flower_measurements = [[5.1, 3.5, 1.4, 0.2]]  # Sepal length, sepal width, petal length, petal width
predicted_class = model.predict(new_flower_measurements)
print("Predicted class:", iris.target_names[predicted_class[0]])
