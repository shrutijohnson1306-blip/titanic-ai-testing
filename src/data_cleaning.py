import pandas as pd

def clean_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=['Name', 'Ticket', 'Cabin'], errors='ignore')
    
    # Fill missing Age and Fare with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Fill Embarked with the most frequent value
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    return df

