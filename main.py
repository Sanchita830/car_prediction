import pandas as pd

from eda import load_data, formatting
from preprocessing import cleaned_data
from visualization import visualize
from training import train_model, predict_price

RAW_FILE = "imports-85.data"
CLEAN_FILE = "cleaned_data.csv"


def run_eda():

    df = pd.read_csv(RAW_FILE, header=None)

    print("\nRunning EDA...\n")

    load_data(df)

    df = formatting(df)

    df.to_csv("formatted_data.csv", index=False)

    print("\nEDA Completed Successfully")


def run_preprocessing():

    print("\nRunning Preprocessing...\n")

    df = pd.read_csv("formatted_data.csv")

    df = cleaned_data(df)

    df.to_csv(CLEAN_FILE, index=False)

    print("\nPreprocessing Completed Successfully")


def run_visualization():

    print("\nRunning Visualization...\n")

    df = pd.read_csv(CLEAN_FILE)

    visualize(df)

    print("\nVisualization Completed Successfully")

def run_training():

    print("\nTraining Model...")

    model, feature_names = train_model()

    print("\nTraining Completed")

    choice = input(
        "\nDo you want to predict a car price? (y/n): "
    )

    if choice.lower() == "y":

        predict_price(
            model,
            feature_names
        )


def main():

    while True:

        print("\n" + "="*40)
        print("CAR PRICE PREDICTION PROJECT")
        print("="*40)

        print("1. Run EDA")
        print("2. Run Preprocessing")
        print("3. Run Visualization")
        print("4. Run Training")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ")

        if choice == "1":
            run_eda()

        elif choice == "2":
            run_preprocessing()

        elif choice == "3":
            run_visualization()

        elif choice == "4":
            run_training()

        elif choice == "5":
            print("Exiting Project...")
            break
        else:
            print("Invalid Choice. Please enter 1-4.")


if __name__ == "__main__":
    main()