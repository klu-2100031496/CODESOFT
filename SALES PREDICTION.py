# "C:/Users/19125/OneDrive/Documents/Internships/CodSoft/advertising.csv"

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data (replace 'your_data.csv' with your actual file path)
data = pd.read_csv("C:/Users/19125/OneDrive/Documents/Internships/CodSoft/advertising.csv")

# Separate features (X) and target variable (y)
X = data.drop('Sales', axis=1)  # Replace 'sales' with your actual sales column name
y = data['Sales']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate model performance (e.g., Mean Squared Error)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Make a new prediction (replace values with your actual data)
new_data = pd.DataFrame({
    # Replace with your feature names and values
    'TV': [23.8],
    'Radio': [35.1],
    'Newspaper':[65.9],
    # ...
})
predicted_sales = model.predict(new_data)
print(f"Predicted sales for new data: {predicted_sales[0]}")
