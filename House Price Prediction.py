import os
import pandas as pd
DATA_PATH = "data/train.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found. Please download from Kaggle (see README)."
    )
data = pd.read_csv(DATA_PATH)

X = data[['sqft', 'bedrooms', 'bathrooms']]
y = data['price']
