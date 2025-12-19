#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data():
    """
    Loads the Titanic dataset and handles missing values.
    """
    # Load dataset
    print("Loading dataset...")
    df = sns.load_dataset('titanic')

    # Drop columns with excessive missing data or redundancy
    # 'deck' has too many NaN; 'embark_town' and 'who' are redundant
    cols_to_drop = ['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male']
    df = df.drop(columns=cols_to_drop, axis=1)

    # Impute missing values
    # Fill Age with Median (to handle outliers)
    df['age'] = df['age'].fillna(df['age'].median())
    # Fill Embarked with Mode (most common value)
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

    return df

def preprocess_features(df):
    """
    Encodes categorical features into numerical format.
    """
    le = LabelEncoder()

    # Encode 'sex': male=1, female=0 (or vice versa depending on encoding)
    df['sex'] = le.fit_transform(df['sex'])
    
    # Encode 'embarked': S=2, C=0, Q=1
    df['embarked'] = le.fit_transform(df['embarked'])

    return df

def train_and_evaluate(df):
    """
    Splits data, trains the model, and prints evaluation metrics.
    """
    # Define Features (X) and Target (y)
    feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    X = df[feature_cols]
    y = df['survived']

    # Split: 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("\n--- Model Performance Evaluation ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

if __name__ == "__main__":
    # Execution Pipeline
    data = load_and_clean_data()
    data = preprocess_features(data)
    train_and_evaluate(data)


# In[ ]:




