"""
Train/re-create the ANN model used in the notebook.

Put Grocery_Dataset.csv in the same folder as this script, then run:
    python train_model.py

This creates:
    grocery_ann_model.keras
    preprocessor.joblib
"""

import joblib
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

DATA_PATH = "Grocery_Dataset.csv"

df = pd.read_csv(DATA_PATH)

df = (
    df.drop(columns=["Price"], errors="ignore")
      .dropna(
          subset=[
              "Stock",
              "Reorder_Level",
              "Reorder_Quantity",
              "Catagory",
          ]
      )
      .copy()
)

df["Needs_Reorder"] = (df["Stock"] <= df["Reorder_Level"]).astype(int)

X = df[
    [
        "Catagory",
        "Stock",
        "Reorder_Level",
        "Reorder_Quantity",
    ]
]
y = df["Needs_Reorder"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            ["Stock", "Reorder_Level", "Reorder_Quantity"],
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            ["Catagory"],
        ),
    ]
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

model = Sequential(
    [
        Input(shape=(X_train_proc.shape[1],)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    X_train_proc,
    y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
)

loss, accuracy = model.evaluate(X_test_proc, y_test, verbose=0)
print(f"Test accuracy: {accuracy * 100:.2f}%")

model.save("grocery_ann_model.keras")
joblib.dump(preprocessor, "preprocessor.joblib")

print("Saved grocery_ann_model.keras")
print("Saved preprocessor.joblib")
