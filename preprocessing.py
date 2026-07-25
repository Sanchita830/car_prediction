import pandas as pd
import numpy as np


def cleaned_data(df):

    print("Missing Values:")
    print(df.isnull().sum())

    # Replace ? with NaN
    df.replace("?", np.nan, inplace=True)

    # Convert required columns to numeric
    cols = [
        "bore",
        "normalized-losses",
        "price",
        "peak-rpm",
        "stroke",
        "horsepower",
        "wheel-base",
        "width",
        "length",
        "height",
        "curb-weight",
        "engine-size",
        "city-mpg",
        "highway-mpg"
    ]

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing values

    df["normalized-losses"] = df["normalized-losses"].fillna(
        df["normalized-losses"].mean()
    )

    df["bore"] = df["bore"].fillna(
        df["bore"].mean()
    )

    df["stroke"] = df["stroke"].fillna(
        df["stroke"].mean()
    )

    df["horsepower"] = df["horsepower"].fillna(
        df["horsepower"].mean()
    )

    df["peak-rpm"] = df["peak-rpm"].fillna(
        df["peak-rpm"].mean()
    )

    df["price"] = df["price"].fillna(
        df["price"].mean()
    )

    df["num-of-doors"] = df["num-of-doors"].fillna(
        df["num-of-doors"].mode()[0]
    )

    # Fuel Consumption Features

    df["city-L/100km"] = 235 / df["city-mpg"]

    df["highway-L/100km"] = 235 / df["highway-mpg"]

    # Feature Engineering

    df["power-to-weight ratio"] = (
        df["horsepower"] / df["curb-weight"]
    )

    df["engine-size-to-weight ratio"] = (
        df["engine-size"] / df["curb-weight"]
    )

    df["length-to-width ratio"] = (
        df["length"] / df["width"]
    )

    df["height-to-width ratio"] = (
        df["height"] / df["width"]
    )

    df["wheel-base-to-length ratio"] = (
        df["wheel-base"] / df["length"]
    )

    df["engine-size-to-wheelbase ratio"] = (
        df["engine-size"] / df["wheel-base"]
    )

    df["horsepower-to-engine size ratio"] = (
        df["horsepower"] / df["engine-size"]
    )

    # Drop old MPG columns

    df.drop(
        ["city-mpg", "highway-mpg"],
        axis=1,
        inplace=True
    )

    # One-Hot Encoding

    dummy_variable_1 = pd.get_dummies(
        df["fuel-type"],
        prefix="fuel_type"
    )

    df = pd.concat(
        [df, dummy_variable_1],
        axis=1
    )

    df.drop(
        "fuel-type",
        axis=1,
        inplace=True
    )

    # Reset Index

    df.reset_index(
        drop=True,
        inplace=True
    )

    print("\nFinal Dataset Shape:")
    print(df.shape)

    print("\nRemaining Missing Values:")
    print(df.isnull().sum().sum())

    # Save cleaned dataset

    df.to_csv(
        "cleaned_data.csv",
        index=False
    )

    print("\nCleaned dataset saved successfully.")

    return df