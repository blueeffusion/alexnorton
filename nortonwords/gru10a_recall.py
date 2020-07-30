import os
import tempfile
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import h5py

path_to_file = tf.keras.utils.get_file('nudata.txt', 'https://nilsgibson.com/txt/nudata.txt')
# Read and decode for python2 compatibility.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# Create set of unique characters in the text
vocab = sorted(set(text))
# two maps, characters to indices and reverse
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)
# "from tensor slices" converts the text vector into a stream of character indices
# chunks are seq_length+1
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
# dupe sequence, shift by 1 to make input and target
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# make training batches
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset

# BUILD keras.sequential three layer model: input(embedding), GRU, output(dense)
# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                    batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

model.summary()

# start TRAINING model
# loss function
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# configure training
model.compile(optimizer='adam', loss=loss)

# set up checkpoints to save during training
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    verbose=1)

# starting with few epochs to minimize calculation time
# EPOCHS=1

# history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

#GENERATE TEXT
# Because of the way the RNN state is passed from timestep to timestep,
# the model only accepts a fixed batch size once built.
# To run the model with a different batch_size, we need to
# rebuild the model and restore the weights from the checkpoint.

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

# latest = tf.train.latest_checkpoint(checkpoint_dir)
# latest
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
    num_generate = 1500

  # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
    text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
    temperature = 0.7

  # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
      # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

with open("output.txt", "w") as text_file:
    print(generate_text(model, start_string=u"Alex says: "), file=text_file)

# print(generate_text(model, start_string=u"Alex says: "))
