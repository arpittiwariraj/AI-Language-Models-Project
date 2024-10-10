# train.py
import tensorflow as tf
from transformer_model import Transformer
from utils.data_preprocessing import load_data

# Load data
train_data, val_data = load_data("data/train.csv", "data/val.csv")

# Initialize the model
model = Transformer(num_layers=2, d_model=128, num_heads=4, dff=512, input_vocab_size=8500, target_vocab_size=8500)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save the trained model
model.save('models/language_model.h5')
