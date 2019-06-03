from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers

import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

history = network.fit(train_images, train_labels, epochs=20, batch_size=128)

history_dict = history.history
print(history_dict)
loss = history_dict['loss']
acc = history_dict['acc']

epochs = range(1, len(loss) + 1)

test_loss, test_acc = network.evaluate(test_images, test_labels)

plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epochs, acc, 'r', label="Validation loss")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
