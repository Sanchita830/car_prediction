import gradio as gr
import pandas as pd
import joblib

from visualization import visualize

# Load trained model
model = joblib.load("car_price_model.pkl")

# Load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Keep only numeric columns (same as training)
df = df.select_dtypes(include=["int64", "float64", "bool"])

# Convert booleans to float
df = df.astype(float)

# Get feature names used during training
feature_names = list(df.drop("price", axis=1).columns)


def predict_car_price(*inputs):
    """
    Predict car price
    """

    input_df = pd.DataFrame(
        [inputs],
        columns=feature_names
    )

    prediction = model.predict(input_df)[0]

    return f"₹ {prediction:,.2f}"


def show_visualization():
    """
    Display visualization
    """

    data = pd.read_csv("cleaned_data.csv")

    return visualize(data)


with gr.Blocks(title="Car Price Prediction Dashboard") as app:

    gr.Markdown(
        """
        # 🚗 Car Price Prediction Dashboard

        Welcome to the Machine Learning Car Price Prediction System
        """
    )

    with gr.Tab("Prediction"):

        gr.Markdown("## Enter Car Details")

        input_components = []

        for feature in feature_names:
            component = gr.Number(
                label=feature,
                value=0
            )
            input_components.append(component)

        predict_button = gr.Button(
            "Predict Price"
        )

        prediction_output = gr.Textbox(
            label="Predicted Price"
        )

        predict_button.click(
            fn=predict_car_price,
            inputs=input_components,
            outputs=prediction_output
        )

    with gr.Tab("Visualization"):

        visualize_button = gr.Button(
            "Generate Visualization"
        )

        plot_output = gr.Plot()

        visualize_button.click(
            fn=show_visualization,
            outputs=plot_output
        )


if __name__ == "__main__":
    app.launch()