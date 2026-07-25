#importing dependencies
import pandas as pd
import numpy as np

#importing the dataset
df = pd.read_csv(r'C:\Users\S8861822\Desktop\Car_prediction\imports-85.data')

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

