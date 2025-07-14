import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)

    # Fill missing values
    df.fillna(0, inplace=True)

    # Encode categorical variables
    for col in df.select_dtypes(include='object').columns:
        if col != 'customerID':  # skip unique IDs
            df[col] = LabelEncoder().fit_transform(df[col])
    
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess('data/customer_data.csv', 'data/processed_data.csv')
