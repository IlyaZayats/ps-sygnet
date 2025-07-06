import numpy as np
import tensorflow as tf
from tensorflow import keras
from server.models.cnn import get_model
from skimage.transform import resize

class ClassificationWeights:
    def __init__(self):
        self.model = get_model(192,192,128)
        self.model.load_weights("sa_192x192x128_10.h5")


def make_heatmap(img_array, model, last_conv_layer_name, pred_index=None):

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = np.zeros(output.shape[0:3], dtype=np.float32)

    for index, w in enumerate(weights):
        cam += w * output[:, :, :, index]

    cam = np.array(cam)
    capi = resize(cam, (192, 192, 128))
    capi = np.maximum(capi,0)
    heatmap = (capi - capi.min()) / (capi.max() - capi.min())

    return heatmap

def get_grad_cam(volume, model):
    last_conv_layer_name = "conv3d_3"
    print("Making Grad-CAM...")
    print(volume.shape)
    return make_heatmap(volume, model, last_conv_layer_name)

def get_prediction(volume, model):
    print("Making Prediction...")
    prediction = model.predict(np.expand_dims(volume, axis=0))[0]
    return prediction