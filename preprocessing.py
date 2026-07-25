import pandas as pd
import numpy as np

def count_missing_values(df):
    # Counting missing values in each column
    missing_values = df.isnull().sum()
    print("Missing values in each column:")
    print(missing_values)

def cleaning(df):
    # Handling missing values
    df = df.dropna()

    # Converting data types
    df[["bore"]] = df[["bore"]].astype("float")
    df[["normalized-losses"]] = df[["normalized-losses"]].astype("float")
    df[["price"]] = df[["price"]].astype("float")
    df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
    df[["stroke"]]=df[["stroke"]].astype("float")
    df[["horsepower"]]=df[["horsepower"]].astype("float")

    #replacing nan values with mean
    df["normalized-losses"].replace(np.nan, df["normalized-losses"].mean(), inplace=True)
    df["bore"].replace(np.nan,np.mean(df["bore"]),inplace=True)
    df["stroke"].replace(np.nan, np.mean(df["stroke"]), inplace=True)
    df["horsepower"].replace(np.nan, np.mean(df["horsepower"]), inplace=True)
    df["peak-rpm"].replace(np.nan, np.mean(df["peak-rpm"]), inplace=True)
    df["price"].replace(np.nan, np.mean(df["price"]), inplace=True)
    df["num-of-doors"].replace(np.nan, "four", inplace=True)
    df["num-of-doors"].replace(np.nan, df["num-of-doors"].value_counts().idxmax(), inplace=True)
    df["num-of-doors"].replace(np.nan, df["num-of-doors"].mode()[0], inplace=True)

    #printing the cleaned dataset
    print("Cleaned dataset:")
    print(df.head(5))

    #showing missing values after cleaning of entire dataset
    print("Missing values after cleaning:") 
    print(df.isnull().sum().sum())

    #reseting the index of the dataset
    df.reset_index(drop=True, inplace=True)

    #saving the cleaned dataset to a new CSV file
    df.to_csv(r'C:\Users\S8861822\Desktop\Car_prediction\cleaned_data.csv', index=False)
