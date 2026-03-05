import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# load dataset
data = pd.read_csv("dataset.csv")

# remove extra spaces from column names
data.columns = data.columns.str.strip()

# print columns to verify
print("Dataset Columns:", data.columns)

# features and target
X = data.drop("target", axis=1)
y = data["target"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create model
model = RandomForestClassifier()

# train model
model.fit(X_train, y_train)

# save model
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")
