#importing dependencies
import pandas as pd
import numpy as np

#performing operations on the dataset
def load_data(df):
    #printing the first 5 rows of the dataset
    print("First 5 rows of the dataset:")
    print(df.head(5))

    #printing the shape of the dataset
    print("Shape of the dataset:")
    print(df.shape)

    #printing the statistics of the dataset
    print("Statistics of the dataset:")
    print(df.describe())

    #printing the information of the dataset
    print("Information of the dataset:")    
    print(df.info())

def formatting(df):
    #remnaming the columns of the dataset
    headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
    df.columns = headers

    #printing data types of the columns
    print("Data types of the columns:")
    print(df.dtypes)

    return df