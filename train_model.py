import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('data/processed_data.csv')
X = df.drop(['churn', 'customerID'], axis=1, errors='ignore')
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'model/model.pkl')
print("✅ Model trained and saved.")
