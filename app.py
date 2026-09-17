import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Grocery Reorder Predictor",
    page_icon="🛒",
    layout="centered",
)

MODEL_PATH = Path("grocery_ann_model.keras")
PREPROCESSOR_PATH = Path("preprocessor.joblib")


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


model, preprocessor = load_artifacts()

st.title("🛒 Grocery Reorder Predictor")
st.caption("ANN-based inventory prediction")
st.write(
    "Enter the current inventory details to predict whether the item "
    "needs to be reordered."
)

if model is None or preprocessor is None:
    st.error("Model files are missing.")
    st.markdown(
        """
        Add these two files to the root of the GitHub repository:

        - `grocery_ann_model.keras`
        - `preprocessor.joblib`

        Then push the files to GitHub and redeploy the Streamlit app.
        """
    )
    st.stop()

st.divider()

# The notebook uses the exact column name "Catagory".
category = st.text_input(
    "Category",
    placeholder="Example: Beverages",
)

stock = st.number_input(
    "Current Stock",
    min_value=0.0,
    value=10.0,
    step=1.0,
)

reorder_level = st.number_input(
    "Reorder Level",
    min_value=0.0,
    value=50.0,
    step=1.0,
)

reorder_quantity = st.number_input(
    "Reorder Quantity",
    min_value=0.0,
    value=40.0,
    step=1.0,
)

if st.button("🔮 Predict", type="primary", use_container_width=True):
    if not category.strip():
        st.warning("Please enter a category.")
        st.stop()

    sample = pd.DataFrame(
        [
            {
                "Catagory": category.strip(),
                "Stock": stock,
                "Reorder_Level": reorder_level,
                "Reorder_Quantity": reorder_quantity,
            }
        ]
    )

    try:
        sample_processed = preprocessor.transform(sample)
        probability = float(model.predict(sample_processed, verbose=0)[0][0])

        # Binary ANN output: 1 = reorder, 0 = sufficient.
        if probability >= 0.5:
            confidence = probability * 100
            st.error("⚠️ REORDER NEEDED")
        else:
            confidence = (1 - probability) * 100
            st.success("✅ STOCK SUFFICIENT")

        st.metric("Prediction Confidence", f"{confidence:.2f}%")

        with st.expander("Input details"):
            st.dataframe(sample, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error("Prediction failed.")
        st.exception(exc)

st.divider()
st.caption("Model: TensorFlow/Keras ANN • Preprocessing: StandardScaler + OneHotEncoder")
