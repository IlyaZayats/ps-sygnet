import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend


class Scalar(layers.Layer):
    def build(self, input_shape):
        self._x = tf.Variable(0.0, dtype=float, trainable=True)
        self._trainable_weights = [self._x]

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return inputs * self._x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class Model:
    def __init__(self, height, width, depth, att=False):
        self.height = height
        self.width = width
        self.depth = depth
        self.att = att

    def get_model(self):

        inputs = keras.Input((self.width, self.height, self.depth, 1))
        filters = 64
        skip_conn = list()

        x = self.conv_bn_relu(inputs, filters, (7, 7, 7))
        skip_conn.append(x)
        x = layers.MaxPool3D(pool_size=(2, 2, 2), padding='same')(x)

        blocks = [3, 4, 6]
        for i in range(len(blocks)):
            if i != 0:
                filters *= 2
            for _ in range(blocks[i]):
                x = self.res_conv_block(x, filters)
            skip_conn.append(x)
            x = layers.MaxPool3D(pool_size=(2, 2, 2), padding='same')(x)

        filters *= 2
        x = self.conv_bn_relu(x, filters, (3, 3, 3))

        s = list()
        s.append(self.conv_bn_relu(x, filters, (1, 1, 1)))
        s.append(self.conv_bn_relu_dilation(x, filters, (3, 3, 3), (2, 2, 2)))
        s.append(self.conv_bn_relu_dilation(x, filters, (3, 3, 3), (3, 3, 3)))
        s.append(self.conv_bn_relu(x, filters, (1, 1, 1)))

        s[-1] = layers.AveragePooling3D(pool_size=1, padding='same')(s[-1])
        x = keras.layers.add(s)
        # x2 = keras.layers.add([s[2], s[3]])
        # x = keras.layers.add([x1, x2])
        # x = tf.math.add_n(s)

        x = self.conv_bn_relu(x, filters, (3, 3, 3))

        filters //= 2
        x = layers.UpSampling3D(2)(x)
        x = self.decoder_block(x, filters)

        for i in range(2, -1, -1):
            filters //= 2
            x = layers.UpSampling3D(2)(x)
            x = tf.concat([x, skip_conn[i]], axis=4)
            # x = keras.ops.concatenate([x, skip_conn[i]], axis=4)
            # print(tf.shape(x)[-1])
            x = self.decoder_block(x, filters)

        outputs = layers.Conv3D(1, (3, 3, 3), activation="sigmoid", padding='same')(x)
        return keras.Model(inputs, outputs, name="SygNet")

    def conv_reshape(self, inputs, num_filters):
        conv = layers.Conv3D(filters=num_filters, kernel_size=(1, 1, 1), padding="same")(inputs)
        return layers.Reshape((conv.shape[1] * conv.shape[2] * conv.shape[3], conv.shape[4]))(conv)

    def self_attention(self, inputs, ratio=2):
        f = self.conv_reshape(inputs, (inputs.shape[4]) // ratio)
        g = self.conv_reshape(inputs, (inputs.shape[4]) // ratio)
        h = self.conv_reshape(inputs, (inputs.shape[4]) // ratio)
        f_transposed = tf.transpose(f, perm=(0, 2, 1))
        # f_transposed = keras.ops.transpose(f, axes=(0,2,1))
        beta = layers.Dot(axes=(1, 2))([f_transposed, g])
        beta = layers.Softmax()(beta)
        teta = layers.Dot(axes=(2, 1))([beta, h])
        teta = layers.Reshape((inputs.shape[1], inputs.shape[2], inputs.shape[3], teta.shape[2]))(teta)
        teta = layers.Conv3D(filters=inputs.shape[4], kernel_size=(1, 1, 1), padding="same")(teta)
        teta = Scalar()(teta)
        return layers.add([inputs, teta])

    def conv_bn_relu_dilation(self, inputs, filters, kernel_size, dilation_rate):
        x = layers.Conv3D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same")(inputs)
        x = layers.BatchNormalization(axis=4)(x)
        return layers.ReLU()(x)

    def conv_bn_relu_strides(self, inputs, filters, kernel_size, stride):
        x = layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=stride, padding="same")(inputs)
        x = layers.BatchNormalization(axis=4)(x)
        return layers.ReLU()(x)

    def conv_bn_relu(self, inputs, filters, kernel_size):
        x = layers.Conv3D(filters=filters, kernel_size=kernel_size, padding="same")(inputs)
        x = layers.BatchNormalization(axis=4)(x)
        return layers.ReLU()(x)

    def upconv_bn_relu(self, inputs, filters, kernel_size):
        x = layers.Conv3DTranspose(filters=filters, kernel_size=kernel_size, padding="same")(inputs)
        x = layers.BatchNormalization(axis=4)(x)
        return layers.ReLU()(x)

    def res_conv_block(self, inputs, filters):
        shortcut = self.conv_bn_relu(inputs, filters, (1, 1, 1))
        x = [self.conv_bn_relu(inputs, filters // 64, (1, 1, 1)) for _ in range(32)]
        y = [self.conv_bn_relu(x[i], filters // 64, (3, 3, 3)) for i in range(len(x))]
        z = tf.concat(y, axis=4)
        # z = keras.ops.concatenate(y, axis=4)
        z = self.conv_bn_relu(z, filters, (1, 1, 1))
        z = keras.layers.add([shortcut, z])
        # z = keras.layers.MultiHeadAttention(num_heads=1, key_dim=2, value_dim=2)(z, z, z, training=True)
        if self.att:
            z = self.self_attention(z)
        return z

    def decoder_block(self, inputs, filters):
        x = self.conv_bn_relu(inputs, filters, (3, 3, 3))
        x = self.upconv_bn_relu(x, filters, (4, 4, 4))
        x = self.conv_bn_relu(x, filters, (3, 3, 3))
        return x