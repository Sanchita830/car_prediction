import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize(df):

    print("Scatter Plot")

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        x=df["city-L/100km"],
        y=df["price"]
    )

    plt.title("City-L/100km vs Price")
    plt.xlabel("City-L/100km")
    plt.ylabel("Price")

    plt.show()

