import tensorflow as tf
from tensorflow import keras

def custom_network():
    odom_input = keras.Input(shape=(5,), name="odom", dtype=float)
    laser_input = keras.Input(shape=(1080,), name="laser_scan", dtype=float)
    D1 = keras.layers.Dense(5, activation="relu")(odom_input)
    D2 = keras.layers.Dense(5, activation="relu")(D1)
    PC = keras.layers.Reshape((1080,1))(laser_input)
    #Expected output (None, 104, 10)
    #10 Filters, each with kernel size 50, stride 10
    C1 = keras.layers.Conv1D(10, 50, strides=10, activation="relu")(PC)
    #Add trainable specification for testing
    #C1.trainable = False
    D3 = keras.layers.Dense(50, activation="relu")(C1)
    D3F = keras.layers.Flatten()(D3)
    Combined = keras.layers.Concatenate()([D3F, D2])
    D4 = keras.layers.Dense(20, activation="relu")(Combined)
    network = keras.Model([odom_input, laser_input], D4)
    return network