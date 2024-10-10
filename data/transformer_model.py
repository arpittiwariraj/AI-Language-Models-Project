# transformer_model.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, MultiHeadAttention, LayerNormalization

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size):
        super(Transformer, self).__init__()

        self.embedding = Embedding(input_vocab_size, d_model)
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(target_vocab_size)

    def call(self, inputs):
        x = self.embedding(inputs)
        attn_output = self.attention(x, x)
        x = LayerNormalization()(x + attn_output)
        x = self.dense1(x)
        return self.dense2(x)
