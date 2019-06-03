import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import reuters
from keras import models
from keras import layers


def vectorize_sequence(sequence, dimension=10000):
    results = np.zeros((len(sequence), dimension))
    for i, sequence in enumerate(sequence):
        results[i, sequence] = 1.
    return results


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

word_index = reuters.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newsWire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

one_hot_train_label = to_one_hot(train_labels)
one_hot_test_label = to_one_hot(test_labels)

network = models.Sequential()
network.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(46, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['acc'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_label[:1000]
partial_y_train = one_hot_train_label[1000:]

history = network.fit(partial_x_train,
                      partial_y_train,
                      epochs=9,
                      batch_size=512,
                      validation_data=(x_val, y_val))

predictions = network.predict(x_test)
print(predictions)
print(np.argmax(predictions[0]))

result = network.evaluate(x_test, one_hot_test_label)

print(result)

loss = history.history['loss']
val_loss = history.history['val_loss']

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(loss)+1)


plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title("Training and Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.xlabel("Epochs")
plt.ylabel('Acc')
plt.legend()
plt.show()
