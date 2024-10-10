# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(train_path, val_path):
    # Load and preprocess the dataset
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    return train_data, val_data

def preprocess_text(text):
    # Basic text preprocessing
    return text.lower().split()
