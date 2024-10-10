# generate.py
import tensorflow as tf
from transformer_model import Transformer
from utils.data_preprocessing import preprocess_text

# Load the trained model
model = tf.keras.models.load_model('models/language_model.h5')

# Function to generate text
def generate_text(input_text):
    # Preprocess input text
    processed_text = preprocess_text(input_text)
    # Generate output text using the model
    output = model.predict(processed_text)
    return output

# Example use
input_text = "Artificial Intelligence is"
generated_text = generate_text(input_text)
print("Generated Text:", generated_text)
