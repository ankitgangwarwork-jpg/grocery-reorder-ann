# 🛒 Grocery Reorder Prediction — ANN + Streamlit

A simple machine-learning web application that predicts whether a grocery item needs to be reordered.

## Model

The original notebook trains a TensorFlow/Keras Artificial Neural Network (ANN).

### Inputs

- Category (`Catagory`)
- Current stock (`Stock`)
- Reorder level (`Reorder_Level`)
- Reorder quantity (`Reorder_Quantity`)

### Output

- `REORDER NEEDED`
- `STOCK SUFFICIENT`

The training target in the notebook is:

```python
df["Needs_Reorder"] = (df["Stock"] <= df["Reorder_Level"]).astype(int)
```

The preprocessing pipeline uses:

- `StandardScaler` for numeric features
- `OneHotEncoder(handle_unknown="ignore")` for category
- `ColumnTransformer` to combine them

The ANN architecture is:

```text
Input
  ↓
Dense(64, ReLU)
  ↓
Dropout(0.2)
  ↓
Dense(32, ReLU)
  ↓
Dense(16, ReLU)
  ↓
Dense(1, Sigmoid)
```

## Project structure

```text
grocery-reorder-ann/
│
├── app.py
├── grocery_ann_model.keras
├── preprocessor.joblib
├── requirements.txt
├── train_model.py
├── README.md
├── .gitignore
└── _ann.ipynb
```

> `grocery_ann_model.keras` and `preprocessor.joblib` are generated model artifacts. Add the copies produced by your notebook to the repository root.

## Run locally

Create/activate a virtual environment if desired, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

## If you need to recreate the model

Put your original `Grocery_Dataset.csv` beside `train_model.py`:

```text
grocery-reorder-ann/
├── Grocery_Dataset.csv
├── train_model.py
└── ...
```

Then run:

```bash
python train_model.py
```

It will generate:

```text
grocery_ann_model.keras
preprocessor.joblib
```

Do not upload the training CSV unless you want it publicly visible in your GitHub repository.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository, for example:
   `grocery-reorder-ann`

2. Upload/push these files:

```text
app.py
grocery_ann_model.keras
preprocessor.joblib
requirements.txt
README.md
train_model.py
_ann.ipynb
.gitignore
```

3. Go to Streamlit Community Cloud:
   https://share.streamlit.io/

4. Sign in with GitHub and authorize Streamlit.

5. Click **Create app**.

6. Select:
   - Repository: your GitHub repository
   - Branch: `main`
   - Main file path: `app.py`

7. Optionally choose a custom app subdomain.

8. Click **Deploy**.

9. Wait for the build to finish. Streamlit will provide a public `.streamlit.app` URL.

## Updating the live app

After deployment, update `app.py` or another project file and push the commit to GitHub. Streamlit Community Cloud monitors the repository and updates the deployed app.

## Important

The application does not need the original training CSV for prediction. It needs the two saved artifacts:

```text
grocery_ann_model.keras
preprocessor.joblib
```

Keep those files in the same directory as `app.py`.

## Example

For:

```text
Category = Beverages
Current Stock = 10
Reorder Level = 50
Reorder Quantity = 40
```

the model can return:

```text
REORDER NEEDED
```
