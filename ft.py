#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import math
from rnn import RNN

FILE_NAMES = ['pass.txt', 'fail.txt', 'comments.txt']

BUFFER_SIZE = 50000
BATCH_SIZE = 32


def main():
    global encoder
    count = 0
    for filename in FILE_NAMES[:-1]:
        count += file_len('clean/' + filename)
    take_size = math.floor(count / 5)

    tmp = labeled_data()
    data = tmp[-1]
    all_labeled_data = shuffle_labled_data(tmp)
    train_test_data = shuffle_labled_data(tmp[:-1])

    tokenizer = tfds.features.text.Tokenizer()
    vocabulary_set = set()
    for text_tensor, _ in all_labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)
    vocab_size = len(vocabulary_set)
    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

    train_test_data = train_test_data.map(encode_map_fn)
    data = data.map(encode_map_fn)
    data = data.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))
    for ex in data.take(5):
        print(ex)

    train_data = train_test_data.skip(take_size).shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))

    test_data = train_test_data.take(take_size)
    test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))


    vocab_size += 1
    model = RNN(vocab_size, train_data, test_data)

    result = model.predict_classes(
        data, batch_size=None, verbose=0
    )

    for i, v in enumerate(result):
        if v == 0:
            print(i + 1)

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

def to_do_data():
    lines_dataset = tf.data.TextLineDataset('clean/comments.txt')
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, 2))
    lines_dataset = lines_dataset.padded_batch(BATCH_SIZE, padded_shapes=([])).map(encode_map_fn)

#  def text_encode(text_tensor):
  #  encoded_text = encoder.encode(text_tensor.numpy())
  #  return encoded_text
#
#  def text_encode_map_fn(text):
  #  encoded_text = tf.py_function(text_encode,
                                       #  inp=[text],
                                       #  Tout=(tf.int64))
  #  encoded_text.set_shape([None])
  #  return encoded_text
#
#  def try_data():
    #  lines_dataset = tf.data.TextLineDataset('clean/pass.txt')
    #  lines_dataset = lines_dataset.padded_batch(BATCH_SIZE, padded_shapes=([])).map(text_encode_map_fn)


if __name__ == "__main__":
    main()
