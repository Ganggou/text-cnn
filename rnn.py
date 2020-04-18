import tensorflow as tf

def RNN(vocab_size, train_data, test_data):
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

    return model
