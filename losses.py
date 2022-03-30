from tensorflow import keras
from scipy import signal
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.applications import vgg16
import tensorflow_addons as tfa
from config import get_config


config, unparsed = get_config()


def gaussian_kernel(kernel_size, std):
    gkern1d = signal.gaussian(kernel_size, std=std).reshape(kernel_size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d/np.power(kernel_size, 2)

def bce(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(y_true, y_pred, sample_weight = y_true*1.2+0.5)

def blur_mse_loss(y_true, y_pred):
    kernel_size = 7
    std = 1

    kernel = tf.constant(gaussian_kernel(kernel_size=kernel_size, std=std),
                         shape=[kernel_size,
                                kernel_size,
                                1, 1],
                         dtype=tf.float32)

    blurred_y_true = tf.nn.conv2d(y_true, kernel,
                                  strides=(1, 1), padding="SAME")
    blurred_y_pred = tf.nn.conv2d(y_pred, kernel,
                                  strides=(1, 1), padding="SAME")

    l2loss = keras.losses.mean_squared_error(blurred_y_true, blurred_y_pred)

    l1loss = keras.losses.mean_absolute_error(y_pred, tf.zeros_like(y_pred))
    return l2loss + config.lamda*l1loss


def ssim(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def mse_plus_reg(y_true, y_pred):
    l2loss = keras.losses.mean_squared_error(y_true, y_pred)
    l1loss = keras.losses.mean_absolute_error(y_pred, tf.zeros_like(y_pred))
    return l2loss + config.lamda*l1loss

class LossNetwork(tf.keras.models.Model):
    def __init__(self, content_layer = 'block4_conv2'):
        super(LossNetwork, self).__init__()
        vgg = vgg16.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False
        model_outputs = vgg.get_layer(content_layer).output
        self.model = tf.keras.models.Model(vgg.input, model_outputs)
        # mixed precision float32 output
        self.linear = layers.Activation('linear', dtype='float32') 

    def call(self, x):
        # x = vgg16.preprocess_input(x)
        x = tf.repeat(x, 3, axis=3)
        x = self.model(x)
        return self.linear(x)


class MSEContentLoss(keras.losses.Loss):
    def __init__(self) -> None:
        super().__init__()
        self.loss_net = LossNetwork()

    def call(self, y_true, y_pred):
        l2loss = keras.losses.mean_squared_error(y_true, y_pred)
        content_loss = tf.reduce_mean((self.loss_net(y_true)-self.loss_net(y_pred))**2)
        return l2loss + config.lamda*content_loss
    
def focal_loss(y_true, y_pred):
    loss = tfa.losses.sigmoid_focal_crossentropy(y_true, y_pred)
    return loss

def get_loss():
    if config.loss == "blur":
        return blur_mse_loss

    elif config.loss == "mse-r":
        return mse_plus_reg
    
    elif config.loss == "bce":
        return bce

    elif config.loss == "mse-c":
        mse_content_loss = MSEContentLoss()
        return mse_content_loss
    
    elif config.loss == "focal":
        return focal_loss

    else:
        print("Assuming that you are using a predefined loss")
        return config.loss