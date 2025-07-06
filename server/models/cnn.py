import tensorflow as tf
from tensorflow import keras
from keras import layers

class Scalar(layers.Layer):
    def build(self, input_shape):
        self._x = tf.Variable(0.0, dtype=float, trainable=True)
        self._trainable_weights = [self._x]

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return inputs * self._x

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def conv_reshape(inputs, num_filters):
    conv = layers.Conv3D(filters=num_filters, kernel_size=(1, 1, 1), padding="same")(inputs)
    outputs = layers.Reshape((int(conv.shape[1] * conv.shape[2] * conv.shape[3]), int(conv.shape[4])))(conv)
    return outputs


def self_attention(inputs, dec):
    f = conv_reshape(inputs, int(inputs.shape[4]/dec))
    g = conv_reshape(inputs, int(inputs.shape[4]/dec))
    h = conv_reshape(inputs, int(inputs.shape[4]/dec))
    f_transposed = keras.ops.transpose(f, (0, 2, 1))
    #f_transposed = tf.transpose(f, perm=(0, 2, 1))
    beta = layers.Dot(axes=(1, 2))([f_transposed, g])
    beta = layers.Softmax()(beta)
    teta = layers.Dot(axes=(2, 1))([beta, h])
    teta = layers.Reshape((inputs.shape[1], inputs.shape[2], inputs.shape[3], teta.shape[2]))(teta)
    teta = layers.Conv3D(filters=inputs.shape[4], kernel_size=(1, 1, 1), padding="same")(teta)
    teta = Scalar()(teta)
    return keras.layers.add([inputs, teta])


def get_model(width, height, depth):

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = self_attention(x, 2)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3dcnn")
    return model