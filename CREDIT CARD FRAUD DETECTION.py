import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
#"C:/Users/19125/OneDrive/Documents/Internships/CodSoft/creditcard.csv"
data = pd.read_csv("C:/Users/19125/OneDrive/Documents/Internships/CodSoft/creditcard.csv")

# Preprocessing
# Assuming 'Amount' is the numerical feature to be normalized
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# Assuming 'Class' is the target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Handling Class Imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Training Classification Algorithm
# Using Logistic Regression as an example
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluating Model Performance
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
