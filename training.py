import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


def train_model():

    df = pd.read_csv("cleaned_data.csv")

    df = df.select_dtypes(
        include=["int64", "float64", "bool"]
    )

    df = df.astype(float)

    y = df["price"]

    X = df.drop("price", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nMODEL EVALUATION")
    print("-" * 30)

    print(
        "R2 Score:",
        r2_score(y_test, predictions)
    )

    print(
        "MAE:",
        mean_absolute_error(y_test, predictions)
    )

    print(
        "RMSE:",
        mean_squared_error(
            y_test,
            predictions
        ) ** 0.5
    )

    joblib.dump(
        model,
        "car_price_model.pkl"
    )

    print("\nModel Saved Successfully")

    return model, X.columns


def predict_price(model, feature_names):

    user_inputs = []

    print("\nEnter Car Details")

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

    print(f"\nPredicted Price: ₹ {predicted_price:,.2f}")