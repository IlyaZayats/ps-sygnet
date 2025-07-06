from server.models.sygnet import Model
import numpy as np
import tensorflow as tf

class SygmentationWeights:
    def __init__(self):
        self.model = Model(192,192,16).get_model()
        self.model.load_weights("sygnet_e50_p12-8_d192x192x16_b6_Adam.h5")

def get_heatmap(x, model):
    output = model.predict(np.expand_dims(x[:, :, 0:16], axis=0))[0]
    b = 32
    offset = 0
    print(x.shape[-1])
    print()
    while True:
        print(b)
        print(offset)
        print()
        prediction = model.predict(np.expand_dims(x[:, :, b - 16:b], axis=0))[0]
        if offset != 0:
            output = tf.concat([output, prediction[:, :, 16 - offset:]], axis=2)
            break
        else:
            output = tf.concat([output, prediction], axis=2)
        if b + 16 > x.shape[-1]:
            offset = x.shape[-1] - b
            b = x.shape[-1]
        else:
            b += 16
    return output