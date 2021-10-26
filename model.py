from tensorflow import keras
from tensorflow.keras import layers

# https://github.com/karolzak/keras-unet/blob/master/keras_unet/models/custom_unet.py

def attention_gate(inp_1, inp_2, n_intermediate_filters):
    inp_1_conv = layers.Conv2D(n_intermediate_filters,
                               kernel_size=(1,1),
                               strides=(1,1),
                               padding="same",
                               kernel_initializer="he_normal")(inp_1)
    inp_2_conv = layers.Conv2D(n_intermediate_filters,
                               kernel_size=(1,1),
                               strides=(1,1),
                               padding="same",
                               kernel_initializer="he_normal")(inp_2)
    f = layers.Activation("relu")(layers.add([inp_1_conv, inp_2_conv]))
    g = layers.Conv2D(filters=1,
                      kernel_size=(1,1),
                      strides=(1,1),
                      padding="same",
                      kernel_initializer="he_normal")(f)
    h = layers.Activation("sigmoid")(g)
    return layers.multiply([inp_1, h])


def attention_concat(conv_below, skip_connection):
    below_filters = conv_below.get_shape().as_list()[-1]
    attention_across = attention_gate(skip_connection, conv_below, below_filters)
    return layers.concatenate([conv_below, attention_across])


def conv2d_block(inputs, filters,
                 use_batch_norm=True,
                 kernel_size=(3,3),
                 activation="swish",
                 kernel_initializer="he_normal",
                 padding="same"):
    c = layers.Conv2D(filters,
                      kernel_size,
                      padding=padding,
                      activation=activation,
                      kernel_initializer=kernel_initializer)(inputs)
    if use_batch_norm:
        c = layers.BatchNormalization()(c)
    c = layers.Conv2D(filters,
                      kernel_size,
                      padding=padding,
                      activation=activation,
                      kernel_initializer=kernel_initializer)(c)
    if use_batch_norm:
        c = layers.BatchNormalization()(c)
    return c


def get_model(img_size,
              n_out_channels,
              filters=32,
              num_layers=4,
              use_batch_norm=True):
    inputs = keras.Input(shape=img_size + (21,))
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm)
        down_layers.append(x)
        x = layers.MaxPooling2D(2)(x)
        filters = filters * 2

    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm)

    for conv in reversed(down_layers):
        filters //= 2
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
        x = attention_concat(conv_below=x, skip_connection=conv)
        # x = layers.concatenate([x, conv], axis=3)
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm)

    outputs = layers.Conv2D(n_out_channels, (1, 1), activation="relu")(x)
    
    model = keras.Model(inputs, outputs)
    return model