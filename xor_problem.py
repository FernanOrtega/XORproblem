import random
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def generate_dataset(option='fixed'):
    if option == 'fixed':
        data = np.random.randint(2, size=(100000, 50)).astype('float32')
        labels = [0 if sum(i) % 2 == 0 else 1 for i in data]
    elif option == 'rand':
        data = []
        labels = []
        for i in range(100000):
            # Choose random length
            length = np.random.randint(1, 51)
            data.append(np.random.randint(2, size=(length)).astype('float32'))
            labels.append(0 if sum(data[i]) % 2 == 0 else 1)
        data = np.asarray(data)

    return data, np.asarray(labels, dtype='float32')


def build_model():
    model = Sequential()
    model.add(LSTM(2, input_shape=(50, 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile('adam', loss='binary_crossentropy', metrics=['acc'])

    return model


def split_dataset(data, labels, padding=None):

    if padding is not None:
        data = pad_sequences(data, maxlen=50, dtype='float32', padding=padding)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=7)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    preds = np.round(preds[:, 0]).astype('float32')
    acc = (np.sum(preds == y_test) / len(y_test)) * 100.
    print('Accuracy: {:.2f}%'.format(acc))

    return acc


def plot_model(history_fixed, history_random_pre, history_random_pos, p_epochs):

    epochs = range(1, p_epochs+1)
    plt.figure()
    plt.plot(epochs, history_fixed.history['loss'], label='Tr loss fixed')
    plt.plot(epochs, history_random_pre.history['loss'], label='Tr loss rand pre')
    plt.plot(epochs, history_random_pos.history['loss'], label='Tr loss rand pos')
    plt.plot(epochs, history_fixed.history['val_loss'], label='Val loss fixed')
    plt.plot(epochs, history_random_pre.history['val_loss'], label='Val loss rand pre')
    plt.plot(epochs, history_random_pos.history['val_loss'], label='Val loss rand pos')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(epochs, history_fixed.history['acc'], label='Tr acc fixed')
    plt.plot(epochs, history_random_pre.history['acc'], label='Tr acc rand pre')
    plt.plot(epochs, history_random_pos.history['acc'], label='Tr acc rand pos')
    plt.plot(epochs, history_fixed.history['val_acc'], label='Val acc fixed')
    plt.plot(epochs, history_random_pre.history['val_acc'], label='Val acc rand pre')
    plt.plot(epochs, history_random_pos.history['val_acc'], label='Val acc rand pos')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    return


def main():
    epochs = 20
    results_acc = []
    # Fixed
    data, labels = generate_dataset(option='fixed')
    model = build_model()
    X_train, X_test, y_train, y_test = split_dataset(data, labels)
    history_fixed = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                        validation_split=0.2, shuffle=False, verbose=2)

    results_acc.append(evaluate_model(model, X_test, y_test))

    # Random size
    data, labels = generate_dataset(option='rand')
    # Pre-padding
    model = build_model()
    X_train, X_test, y_train, y_test = split_dataset(data, labels, padding='pre')
    history_random_pre = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                              validation_split=0.2, shuffle=False, verbose=2)
    results_acc.append(evaluate_model(model, X_test, y_test))
    # Pos-padding
    model = build_model()
    X_train, X_test, y_train, y_test = split_dataset(data, labels, padding='post')
    history_random_pos = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                              validation_split=0.2, shuffle=False, verbose=2)
    results_acc.append(evaluate_model(model, X_test, y_test))

    # Plot model acc and loss
    plot_model(history_fixed, history_random_pre, history_random_pos, epochs)
    return


if __name__ == '__main__':
    main()
