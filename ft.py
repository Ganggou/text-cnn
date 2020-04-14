#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import math

FILE_NAMES = ['pass.txt', 'fail.txt']

BUFFER_SIZE = 50000
BATCH_SIZE = 64


def main():
    global encoder
    count = 0
    for filename in FILE_NAMES:
        count += file_len('clean/' + filename)
    take_size = math.floor(count / 5)

    tmp = labeled_data()
    all_labeled_data = shuffle_labled_data(tmp)

    tokenizer = tfds.features.text.Tokenizer()

    vocabulary_set = set()
    for text_tensor, _ in all_labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)
    vocab_size = len(vocabulary_set)

    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
    all_encoded_data = all_labeled_data.map(encode_map_fn)

    train_data = all_encoded_data.skip(take_size).shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))

    test_data = all_encoded_data.take(take_size)
    test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))

    vocab_size += 1
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 64))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

    for units in [64, 64]:
        model.add(tf.keras.layers.Dense(units, activation='relu'))

# 输出层。第一个参数是标签个数。
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(train_data, epochs=3, validation_data=test_data)
    eval_loss, eval_acc = model.evaluate(test_data)

    print('\nEval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))
    model.save('my_model.h5')


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def labeler(example, index):
  return example, tf.cast(index, tf.int64)

def labeled_data():
    labeled_data_sets = []

    for i, file_name in enumerate(FILE_NAMES):
        lines_dataset = tf.data.TextLineDataset('clean/' + file_name)
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
        labeled_data_sets.append(labeled_dataset)

    return labeled_data_sets

def shuffle_labled_data(labeled_data_sets):
    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

    all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    return all_labeled_data.repeat(4)

def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  # py_func doesn't set the shape of the returned tensors.
  encoded_text, label = tf.py_function(encode, 
                                       inp=[text, label], 
                                       Tout=(tf.int64, tf.int64))

  # `tf.data.Datasets` work best if all components have a shape set
  #  so set the shapes manually: 
  encoded_text.set_shape([None])
  label.set_shape([])

  return encoded_text, label

if __name__ == "__main__":
    main()
