#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import os

FILE_NAMES = ['pass.txt', 'fail.txt']

BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000


def main():
    global encoder
    all_labeled_data = shuffle_labled_data(labeled_data())
    vocabulary_set = get_vocabulary_set(all_labeled_data)
    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
    all_encoded_data = all_labeled_data.map(encode_map_fn)
    for ex in all_encoded_data.take(5):
        print(ex)


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

def get_vocabulary_set(data):
    tokenizer = tfds.features.text.Tokenizer()

    vocabulary_set = set()
    for text_tensor, _ in data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)

    return vocabulary_set

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
