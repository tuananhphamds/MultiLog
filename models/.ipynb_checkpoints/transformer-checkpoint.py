import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.models import Sequential

class TransformerBlock(Layer):
    def __init__(self,
                 embed_size,
                 num_heads,
                 ff_dim,
                 dropout=0.1,
                 **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_size)
        self.ffn = Sequential(
            [
                Dense(ff_dim, activation='relu'),
                Dense(embed_size)
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, x, training, mask=None):
        if mask is not None:
            if training is False:
                mask = tf.constant(mask)
            mask = mask[:, tf.newaxis, tf.newaxis, :]
            attn_output = self.attn(x, x, attention_mask=mask)
        else:
            attn_output = self.attn(x, x)
        drop_output1 = self.dropout1(attn_output, training=training)
        norm_output = self.layernorm1(x + drop_output1)
        ffn_output = self.ffn(norm_output)
        drop_output2 = self.dropout2(ffn_output, training=training)
        return self.layernorm2(norm_output + drop_output2)


class TransformerEncoder(Layer):
    def __init__(self,
                 num_blocks,
                 embed_size,
                 num_heads,
                 ff_dim,
                 window_size,
                 vocab_size,
                 dropout=0.1,
                 **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_blocks = num_blocks
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.dropout = dropout
        self.blocks = [
            TransformerBlock(embed_size=self.embed_size,
                             num_heads=self.num_heads,
                             ff_dim=self.ff_dim,
                             dropout=self.dropout)
            for i in range(self.num_blocks)
        ]

    def call(self, x, mask=None):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, mask=mask)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_blocks': self.num_blocks,
            'embed_size': self.embed_size,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'window_size': self.window_size,
            'dropout': self.dropout,
            'vocab_size': self.vocab_size
            }
        )
        return config