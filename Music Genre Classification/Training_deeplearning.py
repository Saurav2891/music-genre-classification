import os
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data_path = "data_json"

def load_data(data_path):
    print("Data loading\n")
    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Loaded Data")

    return x, y



def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":

    #Loading the data and splitting to train and test
    x, y = load_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    #Model creation and training
    model = tf.keras.Sequential([

            tf.keras.layers.Flatten(input_shape = (x.shape[1],x.shape[2])),

            tf.keras.layers.Dense(512, activation = "relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation = "relu"),
            tf.keras.layers.Dense(64, activation = "relu"),

            tf.keras.layers.Dense(10, activation = "softmax")

    ])

    optimizer = tf.keras.optimizers.Adam(lr = 0.0001)
    model.compile(optimizer = optimizer, loss="sparse_categorical_crossentropy", metrics = ["accuracy"])

    model.summary()


    history = model.fit(x_train,y_train, validation_data = (x_test,y_test), batch_size = 32, epochs = 50)
    #plot_history(history)
    model.save("model_deeplearning.h5")
    print("Saved model to disk")


    
