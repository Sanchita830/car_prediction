import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def train_model():

    # Load cleaned dataset
    df = pd.read_csv("cleaned_data.csv")

    # Keep only numerical columns
    df = df.select_dtypes(include=["int64", "float64", "bool"])

    # Convert boolean columns from one-hot encoding
    df = df.astype(float)

    # Target variable
    y = df["price"]

    # Features
    X = df.drop("price", axis=1)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Training Model
    model = LinearRegression()

    model.fit(X_train, y_train)

    # Prediction on test data
    predictions = model.predict(X_test)

    print("\nMODEL EVALUATION")
    print("-" * 30)

    print("R2 Score :", r2_score(y_test, predictions))

    print("MAE :", mean_absolute_error(y_test, predictions))

    print(
        "RMSE :",
        mean_squared_error(
            y_test,
            predictions
        ) ** 0.5
    )

    # Save Model
    joblib.dump(model, "car_price_model.pkl")

    print("\nModel Saved Successfully")

    return model, X.columns

def predict_price(model, feature_names):

    print("\nENTER CAR DETAILS")

    user_inputs = []

    for feature in feature_names:

        value = float(
            input(f"{feature}: ")
        )

        user_inputs.append(value)

    input_df = pd.DataFrame(
        [user_inputs],
        columns=feature_names
    )

    predicted_price = model.predict(
        input_df
    )

    print(f"\nPredicted Car Price: {predicted_price}")