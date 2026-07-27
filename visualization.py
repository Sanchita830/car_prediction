import seaborn as sns
import matplotlib.pyplot as plt


def visualize(df):

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        x=df["city-L/100km"],
        y=df["price"],
        ax=ax
    )

    ax.set_title("City-L/100km vs Price")
    ax.set_xlabel("City-L/100km")
    ax.set_ylabel("Price")

    return fig