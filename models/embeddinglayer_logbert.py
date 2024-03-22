import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dropout

class PositionalEmbedding(Layer):
    def __init__(self,
                 window_size,
                 embed_size,
                 **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        position_embedding_matrix = self.get_position_encoding(window_size, embed_size)
        self.position_embedding_layer = Embedding(input_dim=window_size, output_dim=embed_size,
                                                  weights=[position_embedding_matrix],
                                                  trainable=False)
        self.window_size = window_size

    def get_position_encoding(self, window_size, d, n=10000):
        P = np.zeros((window_size,d))
        for k in range(window_size):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2 * i / d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P

    def call(self, x):
        position_indices = tf.range(start=0, limit=self.window_size, delta=1)
        return x + self.position_embedding_layer(position_indices)


class LogTemplateEmbedding(Layer):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 **kwargs):
        super(LogTemplateEmbedding, self).__init__(**kwargs)
        self.embed_size = embed_size

        self.trainable_embeddings = Embedding(input_dim=vocab_size,
                                              output_dim=embed_size,
                                              trainable=True)

    def call(self, x):
        trainable_x_embedded = self.trainable_embeddings(x)
        return trainable_x_embedded


class BERTEmbedding(Layer):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 window_size,
                 **kwargs):
        super(BERTEmbedding, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.positional = PositionalEmbedding(window_size=self.window_size,
                                              embed_size=self.embed_size)
        self.log_template = LogTemplateEmbedding(vocab_size=self.vocab_size,
                                                 embed_size=self.embed_size)

    def call(self, x):
        log_embed = self.log_template(x)
        output = self.positional(log_embed)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_size': self.embed_size,
            'vocab_size': self.vocab_size,
            'window_size': self.window_size
        }
        )
        return config
