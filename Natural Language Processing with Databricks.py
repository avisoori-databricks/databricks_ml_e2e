# Databricks notebook source
# MAGIC %md
# MAGIC Import the necessary libraries which are pre-installed in the machine learning runtime

# COMMAND ----------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# COMMAND ----------

# MAGIC %md
# MAGIC Define the model
# MAGIC 
# MAGIC Step 1: Implement a Transformer block as a layer

# COMMAND ----------

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# COMMAND ----------

# MAGIC %md
# MAGIC Step 2. Implement embedding layer 
# MAGIC 
# MAGIC Two seperate embedding layers, one for tokens, one for token index (positions).

# COMMAND ----------

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tensorboard
# MAGIC 
# MAGIC Tensorboard provides a nice UI to visualize the training process of your neural network and can help with debugging! We can define it as a callback.

# COMMAND ----------

# MAGIC %load_ext tensorboard

# COMMAND ----------

log_dir = f"/tmp/lowes"

# COMMAND ----------

#We just cleared out the log directory above in case you re-run this notebook multiple times.
%tensorboard --logdir $log_dir

# COMMAND ----------

from tensorflow.keras.callbacks import TensorBoard

### Here, we set histogram_freq=1 so that we can visualize the distribution of a Tensor over time. 
### It can be helpful to visualize weights and biases and verify that they are changing in an expected way. 
### Refer to the Tensorboard documentation linked above.

tensorboard = TensorBoard(log_dir, histogram_freq=1)

# COMMAND ----------

# MAGIC %md
# MAGIC Download and prepare dataset

# COMMAND ----------

vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

# COMMAND ----------

# MAGIC %md
# MAGIC Step 3. Create classifier model using transformer layer
# MAGIC 
# MAGIC Transformer layer outputs one vector for each time step of our input sequence.
# MAGIC 
# MAGIC Here, we take the mean across all time steps and use a feed forward network on top of it to classify text.

# COMMAND ----------

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# COMMAND ----------

# MAGIC %md
# MAGIC Train and Evaluate

# COMMAND ----------

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val), callbacks=[tensorboard]
)

# COMMAND ----------


