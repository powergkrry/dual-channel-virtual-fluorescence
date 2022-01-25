from tensorflow import keras
from scipy import signal
import tensorflow as tf
import numpy as np
from config import get_config


config, unparsed = get_config()


def gaussian_kernel(kernel_size, std):
    gkern1d = signal.gaussian(kernel_size, std=std).reshape(kernel_size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d/np.power(kernel_size, 2)


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


def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))


def mse_plus_reg(y_true, y_pred):
    l2loss = keras.losses.mean_squared_error(y_true, y_pred)
    l1loss = keras.losses.mean_absolute_error(y_pred, tf.zeros_like(y_pred))
    return l2loss + config.lamda*l1loss


def get_loss():
    if config.loss == "blur":
        return blur_mse_loss
    elif config.loss == "mse-r":
        return mse_plus_reg
    else:
        print("Assuming that you are using a predefined loss")
        return config.loss