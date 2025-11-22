import requests
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(test_size=0.2, random_state=42):
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    folder_path = 'data'
    file_path = os.path.join(folder_path, 'titanic.csv')

    # Check if the Titanic dataset exists in the 'data' folder, otherwise download it
    if not os.path.isfile(file_path):
        print("Titanic dataset file not found. Donwloading...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downlaoded dataset to {file_path}")
        else:
            print("Failed to download dataset.")
    else:
        print(f"Dataset file found at {file_path}")
        
    # Load the Titanic dataset
    df = pd.read_csv(file_path)

    # Basic preprocessing: fill missing values, encode categorical
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    X= df[features]
    y= df['Survived']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test