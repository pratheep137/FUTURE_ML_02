import pandas as pd
import joblib
from sklearn.metrics import classification_report

df = pd.read_csv('data/processed_data.csv')
X = df.drop(['churn', 'customerID'], axis=1, errors='ignore')
y = df['churn']

model = joblib.load('model/model.pkl')
y_pred = model.predict(X)

print(classification_report(y, y_pred))
