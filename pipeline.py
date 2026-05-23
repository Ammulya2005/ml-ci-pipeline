import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

print("Stage 1: Loading dataset...")
df = pd.read_csv("data.csv")

print("Stage 2: Data validation...")
required_columns = ["Survived", "Pclass", "Sex", "Age", "Fare"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

print("Data validation completed successfully")

print("Stage 3: Preprocessing...")
df = df[required_columns].copy()
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

print("Stage 4: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Stage 5: Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Stage 6: Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print("Stage 7: Saving artifacts...")
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/titanic_model.pkl")

with open("artifacts/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}")

print("ML pipeline completed successfully")